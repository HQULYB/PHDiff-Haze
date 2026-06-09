from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tools.hazegen_train_utils import (
    image_to_tensor01,
    list_images,
    pair_key,
    read_rgb,
    resize_shorter_than,
)


def cycle(loader: Iterable):
    """Repeat a dataloader forever without caching previous batches."""
    while True:
        for batch in loader:
            yield batch


def parse_exclude_keywords(exclude_keywords: Sequence[str] | str | None) -> list[str]:
    if exclude_keywords is None:
        return []
    if isinstance(exclude_keywords, str):
        parts = exclude_keywords.split(",")
    else:
        parts = list(exclude_keywords)
    return [part.strip().lower() for part in parts if part and part.strip()]


def _excluded(path: Path, keywords: Sequence[str] | str | None) -> bool:
    parsed = parse_exclude_keywords(keywords)
    if not parsed:
        return False
    text = path.as_posix().lower()
    return any(keyword in text for keyword in parsed)


def _filter_images(paths: list[Path], keywords: Sequence[str] | str | None) -> list[Path]:
    filtered = [path for path in paths if not _excluded(path, keywords)]
    if paths and not filtered:
        raise RuntimeError(f"all images were filtered by keywords: {', '.join(parse_exclude_keywords(keywords))}")
    return filtered


def list_images_filtered(folder: str | Path, exclude_keywords: Sequence[str] | str | None = None) -> list[Path]:
    return _filter_images(list_images(folder), exclude_keywords)


def _legacy_excluded(path: Path, keywords: str) -> bool:
    if not keywords:
        return False
    lowered = path.name.lower()
    return any(keyword.strip().lower() in lowered for keyword in keywords.split(",") if keyword.strip())


def build_depth_lookup(depth_dir: str | Path | None, exclude_name_keywords: Sequence[str] | str | None = None) -> dict[str, Path]:
    """Build a loose filename lookup for depth maps."""
    if not depth_dir:
        return {}
    root = Path(depth_dir)
    if not root.exists():
        return {}
    lookup: dict[str, Path] = {}
    for path in list_images_filtered(root, exclude_name_keywords):
        lookup[path.stem] = path
        lookup.setdefault(pair_key(path), path)
    return lookup


def get_matched_depth_path(image_path: Path, depth_lookup: dict[str, Path], required: bool = False) -> Path | None:
    """Find a depth map matching an image by full stem or paired stem."""
    for key in (image_path.stem, pair_key(image_path)):
        if key in depth_lookup:
            return depth_lookup[key]
    if required:
        raise FileNotFoundError(f"no matched depth for image: {image_path}")
    return None


def read_depth01(path: str | Path) -> np.ndarray:
    """Read a depth image and normalize it to [0, 1]."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"failed to read depth image: {path}")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.float32)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    return depth[..., None]


def depth_to_tensor01(depth: Optional[np.ndarray]) -> torch.Tensor:
    """Convert an HW depth array in [0, 1] to a 1HW tensor."""
    if depth is None:
        return torch.empty(0)
    if depth.ndim == 2:
        depth = depth[..., None]
    return torch.from_numpy(depth.astype(np.float32)).permute(2, 0, 1).contiguous().clamp(0.0, 1.0)


def _paired_random_crop_with_depth(
    clean: np.ndarray,
    hazy: np.ndarray,
    depth: np.ndarray | None,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    clean = resize_shorter_than(clean, crop_size)
    hazy = resize_shorter_than(hazy, crop_size)
    if depth is not None:
        depth = resize_shorter_than(depth, crop_size)

    h = min(clean.shape[0], hazy.shape[0], depth.shape[0] if depth is not None else clean.shape[0])
    w = min(clean.shape[1], hazy.shape[1], depth.shape[1] if depth is not None else clean.shape[1])
    clean = clean[:h, :w]
    hazy = hazy[:h, :w]
    if depth is not None:
        depth = depth[:h, :w]

    if h > crop_size and w > crop_size:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        clean = clean[top : top + crop_size, left : left + crop_size]
        hazy = hazy[top : top + crop_size, left : left + crop_size]
        if depth is not None:
            depth = depth[top : top + crop_size, left : left + crop_size]
    else:
        clean = cv2.resize(clean, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        hazy = cv2.resize(hazy, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        if depth is not None:
            depth = cv2.resize(depth, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    if random.random() < 0.5:
        clean = np.ascontiguousarray(clean[:, ::-1])
        hazy = np.ascontiguousarray(hazy[:, ::-1])
        if depth is not None:
            depth = np.ascontiguousarray(depth[:, ::-1])
    return clean, hazy, depth


def _random_crop_with_depth(
    image: np.ndarray,
    depth: np.ndarray | None,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    image = resize_shorter_than(image, crop_size)
    if depth is not None:
        depth = resize_shorter_than(depth, crop_size)
        h = min(image.shape[0], depth.shape[0])
        w = min(image.shape[1], depth.shape[1])
        image = image[:h, :w]
        depth = depth[:h, :w]
    h, w = image.shape[:2]

    if h > crop_size and w > crop_size:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        image = image[top : top + crop_size, left : left + crop_size]
        if depth is not None:
            depth = depth[top : top + crop_size, left : left + crop_size]
    else:
        image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        if depth is not None:
            depth = cv2.resize(depth, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        if depth is not None:
            depth = np.ascontiguousarray(depth[:, ::-1])
    return image, depth


class SyntheticHazePairDataset(Dataset):
    """Paired clean/hazy dataset used by physical Stage1."""

    def __init__(
        self,
        clean_dir: str,
        hazy_dir: str,
        crop_size: int,
        prompt: str,
        exclude_name_keywords: str = "",
        depth_dir: str = "",
        use_depth_condition: bool = True,
    ):
        self.clean_paths = _filter_images(list_images(clean_dir), exclude_name_keywords)
        self.hazy_paths = _filter_images(list_images(hazy_dir), exclude_name_keywords)
        self.crop_size = crop_size
        self.prompt = prompt
        self.use_depth_condition = use_depth_condition
        self.depth_lookup = build_depth_lookup(depth_dir, exclude_name_keywords)

        clean_lookup = {path.stem: path for path in self.clean_paths}
        clean_lookup.update({pair_key(path): path for path in self.clean_paths})
        self.pairs: list[tuple[Path, Path]] = []
        for hazy_path in self.hazy_paths:
            clean_path = clean_lookup.get(hazy_path.stem) or clean_lookup.get(pair_key(hazy_path))
            if clean_path is not None:
                self.pairs.append((clean_path, hazy_path))
        if not self.pairs:
            raise RuntimeError(f"no clean/hazy pairs found: {clean_dir} <-> {hazy_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        clean_path, hazy_path = self.pairs[index % len(self.pairs)]
        clean = read_rgb(clean_path)
        hazy = read_rgb(hazy_path)
        depth = None
        if self.use_depth_condition:
            depth_path = get_matched_depth_path(clean_path, self.depth_lookup, required=False)
            if depth_path is not None:
                depth = read_depth01(depth_path)

        clean, hazy, depth = _paired_random_crop_with_depth(clean, hazy, depth, self.crop_size)
        clean_t = image_to_tensor01(clean)
        hazy_t = image_to_tensor01(hazy) * 2.0 - 1.0
        depth_t = depth_to_tensor01(depth) if depth is not None else torch.empty(0)
        return hazy_t, clean_t, depth_t, self.prompt, str(clean_path)


class ClearImageDataset(Dataset):
    """Clean-image dataset used by physical Stage2."""

    def __init__(
        self,
        clean_dir: str,
        crop_size: int,
        exclude_name_keywords: str = "",
        depth_dir: str = "",
        use_depth_condition: bool = True,
    ):
        self.clean_paths = _filter_images(list_images(clean_dir), exclude_name_keywords)
        self.crop_size = crop_size
        self.use_depth_condition = use_depth_condition
        self.depth_lookup = build_depth_lookup(depth_dir, exclude_name_keywords)

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, index: int):
        clean_path = self.clean_paths[index % len(self.clean_paths)]
        image = read_rgb(clean_path)
        depth = None
        if self.use_depth_condition:
            depth_path = get_matched_depth_path(clean_path, self.depth_lookup, required=False)
            if depth_path is not None:
                depth = read_depth01(depth_path)
        image, depth = _random_crop_with_depth(image, depth, self.crop_size)
        image_t = image_to_tensor01(image)
        depth_t = depth_to_tensor01(depth) if depth is not None else torch.empty(0)
        return image_t, depth_t


def expected_hint_channels(model, latent_channels: int) -> int:
    first_block = model.controlnet.input_blocks[0][0]
    return int(first_block.weight.shape[1] - latent_channels)


def make_depth_condition(
    depth: torch.Tensor | None,
    spatial_size: tuple[int, int],
    scale: float = 1.0,
    zero_center: bool = False,
    invert: bool = False,
    dropout: float = 0.0,
) -> torch.Tensor | None:
    if depth is None or depth.numel() == 0:
        return None
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    depth = depth.float().clamp(0.0, 1.0)
    if invert:
        depth = 1.0 - depth
    if dropout > 0 and torch.rand((), device=depth.device) < dropout:
        depth = torch.zeros_like(depth)
    depth = F.interpolate(depth, size=spatial_size, mode="bilinear", align_corners=False)
    if zero_center:
        depth = depth - depth.mean(dim=(2, 3), keepdim=True)
    return depth * scale


def prepare_cond(
    model,
    clean: torch.Tensor,
    prompt: Sequence[str],
    depth: torch.Tensor | None = None,
    args=None,
) -> dict[str, torch.Tensor]:
    """Build ControlLDM condition, appending depth only when the model expects it."""
    with torch.no_grad():
        c_txt = model.clip.encode(list(prompt))
        clean_latent = model.vae_encode(clean * 2 - 1, sample=False).contiguous().float()
        hint_channels = expected_hint_channels(model, clean_latent.shape[1])
        if hint_channels == clean_latent.shape[1]:
            c_img = clean_latent
        elif hint_channels == clean_latent.shape[1] + 1:
            depth_cond = make_depth_condition(
                depth,
                clean_latent.shape[-2:],
                scale=getattr(args, "depth_condition_scale", 1.0),
                zero_center=getattr(args, "depth_condition_zero_center", False),
                invert=getattr(args, "invert_depth_condition", False),
                dropout=getattr(args, "depth_condition_dropout", 0.0) if model.training else 0.0,
            )
            if depth_cond is None:
                depth_cond = torch.zeros(
                    clean_latent.shape[0],
                    1,
                    clean_latent.shape[-2],
                    clean_latent.shape[-1],
                    device=clean_latent.device,
                    dtype=clean_latent.dtype,
                )
            else:
                depth_cond = depth_cond.to(device=clean_latent.device, dtype=clean_latent.dtype)
            c_img = torch.cat([clean_latent, depth_cond], dim=1)
        else:
            raise RuntimeError(
                f"unsupported ControlNet hint channels: expected {hint_channels}, "
                f"clean latent has {clean_latent.shape[1]} channels"
            )
    return {"c_txt": c_txt, "c_img": c_img.contiguous().float()}


def latent_sample_shape(model, cond: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    """Return the diffusion latent sample shape matching a ControlNet condition."""
    c_img = cond["c_img"]
    return (c_img.shape[0], int(model.controlnet.in_channels), c_img.shape[-2], c_img.shape[-1])
