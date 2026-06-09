from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tools.hazegen_train_utils import image_to_tensor01, list_images, read_rgb


class AirlightHead(nn.Module):
    """Predict a low-dimensional atmospheric light vector A from the clean image."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, clean: torch.Tensor) -> torch.Tensor:
        # Keep A in a bright atmospheric range without letting it become arbitrary RGB.
        return 0.55 + 0.45 * torch.sigmoid(self.net(clean))


def estimate_airlight(hazy: torch.Tensor, top_percent: float = 0.01) -> torch.Tensor:
    """Estimate global atmospheric light from the brightest hazy pixels.

    hazy: BCHW tensor in [0, 1].
    returns: Bx3 tensor in [0, 1].
    """
    b, c, h, w = hazy.shape
    flat = hazy.flatten(2)
    lum = hazy.mean(dim=1).flatten(1)
    k = max(1, int(h * w * top_percent))
    idx = lum.topk(k, dim=1).indices.unsqueeze(1).expand(-1, c, -1)
    airlight = flat.gather(2, idx).mean(dim=2)
    return airlight.clamp(0.55, 1.0)


def estimate_density_from_pair(
    clean: torch.Tensor,
    hazy_m11: torch.Tensor,
    airlight: Optional[torch.Tensor] = None,
    blur_kernel: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate haze density d=1-t from paired clean/hazy images.

    The estimate is intentionally low-frequency and conservative. It is used as
    the diffusion target, while color is carried by airlight A.
    """
    hazy = (hazy_m11 + 1.0) * 0.5
    if airlight is None:
        airlight = estimate_airlight(hazy)
    a = airlight.view(-1, 3, 1, 1)
    denom = (a - clean).clamp_min(0.08)
    density_rgb = ((hazy - clean) / denom).clamp(0.0, 1.0)
    density = density_rgb.mean(dim=1, keepdim=True)
    density = gaussian_blur(density, blur_kernel).clamp(0.0, 1.0)
    return density, airlight


def density_to_carrier(density: torch.Tensor) -> torch.Tensor:
    """Convert 1-channel density [0,1] into a 3-channel VAE carrier [-1,1]."""
    return density.expand(-1, 3, -1, -1) * 2.0 - 1.0


def carrier_to_density(carrier_m11: torch.Tensor) -> torch.Tensor:
    """Decode a 3-channel VAE carrier back into a single density map [0,1]."""
    return ((carrier_m11.mean(dim=1, keepdim=True) + 1.0) * 0.5).clamp(0.0, 1.0)


def compose_physical_haze(clean: torch.Tensor, density: torch.Tensor, airlight: torch.Tensor) -> torch.Tensor:
    """Physical haze composition: I = J * (1-d) + A * d."""
    a = airlight.view(-1, 3, 1, 1).to(device=clean.device, dtype=clean.dtype)
    density = density.clamp(0.0, 1.0)
    return (clean * (1.0 - density) + a * density).clamp(0.0, 1.0)


def gaussian_blur(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1.0) + 0.8
    radius = kernel_size // 2
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - radius
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = image.shape[1]
    kh = kernel_1d.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    kw = kernel_1d.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    out = F.pad(image, (0, 0, radius, radius), mode="reflect")
    out = F.conv2d(out, kh, groups=channels)
    out = F.pad(out, (radius, radius, 0, 0), mode="reflect")
    return F.conv2d(out, kw, groups=channels)


@torch.no_grad()
def build_airlight_bank(real_hazy_dir: str, max_images: int, device: torch.device) -> torch.Tensor:
    paths = list_images(real_hazy_dir)
    if max_images > 0:
        paths = paths[:max_images]
    values = []
    for path in tqdm(paths, desc="Build real haze airlight bank"):
        image = read_rgb(path)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        tensor = image_to_tensor01(image).unsqueeze(0).to(device)
        values.append(estimate_airlight(tensor).cpu())
    if not values:
        raise RuntimeError(f"no real hazy images found for airlight bank: {real_hazy_dir}")
    return torch.cat(values, dim=0).clamp(0.55, 1.0)


def sample_airlight(bank: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.full((batch_size, 3), 0.85, device=device)
    idx = torch.randint(0, bank.shape[0], (batch_size,), device=device)
    return bank.to(device)[idx]


def save_phys_checkpoint(path: Path, model, airlight_head: AirlightHead, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "controlnet": model.controlnet.state_dict(),
        "airlight_head": airlight_head.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_phys_checkpoint(path: str | Path, model, airlight_head: AirlightHead, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    if "controlnet" in payload:
        model.controlnet.load_state_dict(payload["controlnet"], strict=True)
    else:
        model.controlnet.load_state_dict(payload, strict=True)
    if "airlight_head" in payload:
        airlight_head.load_state_dict(payload["airlight_head"], strict=True)
    return payload
