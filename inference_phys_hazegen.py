from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import cv2
import torch
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

from diffbir.pipeline import pad_to_multiples_of
from diffbir.sampler import SpacedSampler
from diffbir.utils.common import instantiate_from_config
from phys_haze_utils import (
    AirlightHead,
    carrier_to_density,
    compose_physical_haze,
    load_phys_checkpoint,
    sample_airlight,
)
from train_hazegen_residual import build_depth_lookup, get_matched_depth_path, prepare_cond, read_depth01, depth_to_tensor01, latent_sample_shape
from tools.hazegen_train_utils import list_images


def load_sd_model(cfg, sd_path: str, device: torch.device):
    model = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(sd_path, map_location="cpu")["state_dict"]
    model.load_pretrained_sd(sd)
    return model.to(device).eval()


def parse_fixed_airlight(value: str, device: torch.device) -> torch.Tensor:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--fixed_airlight must be like 0.85,0.88,0.9")
    return torch.tensor(parts, device=device).view(1, 3).clamp(0.0, 1.0)


@torch.no_grad()
def main(args) -> None:
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    model = load_sd_model(cfg, args.sd_path, device)
    airlight_head = AirlightHead().to(device).eval()
    payload = load_phys_checkpoint(args.controlnet_path, model, airlight_head, map_location="cpu")

    diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    depth_lookup = build_depth_lookup(args.depth_dir)
    images = list_images(args.input)
    input_root = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    airlight_bank = None
    bank_path = args.airlight_bank or payload.get("airlight_bank", "")
    if args.airlight_source == "bank":
        if not bank_path:
            raise ValueError("--airlight_source bank needs --airlight_bank or a checkpoint with airlight_bank metadata")
        airlight_bank = torch.load(bank_path, map_location="cpu").float().clamp(0.55, 1.0)

    fixed_airlight = parse_fixed_airlight(args.fixed_airlight, device) if args.airlight_source == "fixed" else None
    rescaler = Resize(args.min_size, interpolation=InterpolationMode.BICUBIC, antialias=True)

    for image_path in tqdm(images, desc="PhysHaze inference"):
        out_name = image_path.relative_to(input_root).with_suffix(".png") if input_root.is_dir() else Path(image_path.stem + ".png")
        save_path = output_dir / out_name
        if args.skip_existing and save_path.exists():
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"skip unreadable image: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = to_tensor(image_rgb).unsqueeze(0)
        depth_image = None
        if args.use_depth_condition:
            depth_path = get_matched_depth_path(image_path, depth_lookup, True)
            depth_np = read_depth01(depth_path)
            depth_image = depth_to_tensor01(depth_np).unsqueeze(0)

        _, _, original_h, original_w = image.shape
        if min(original_h, original_w) < args.min_size:
            image = rescaler(image)
            if depth_image is not None:
                depth_image = Resize(args.min_size, interpolation=InterpolationMode.NEAREST, antialias=False)(depth_image)
        _, _, resized_h, resized_w = image.shape

        image = pad_to_multiples_of(image, multiple=64).to(device)
        if depth_image is not None:
            depth_image = pad_to_multiples_of(depth_image, multiple=64).to(device)

        cond = prepare_cond(model, image, [args.prompt], depth_image, args)
        uncond = None
        if args.cfg_scale != 1.0:
            uncond = {"c_img": torch.zeros_like(cond["c_img"]), "c_txt": cond["c_txt"].clone()}
        z = sampler.sample(
            model=model,
            device=device,
            steps=args.steps,
            x_size=latent_sample_shape(model, cond),
            cond=cond,
            uncond=uncond,
            cfg_scale=args.cfg_scale,
            progress=False,
        )
        density = carrier_to_density(model.vae_decode(z))
        if args.airlight_source == "head":
            airlight = airlight_head(image)
        elif args.airlight_source == "bank":
            airlight = sample_airlight(airlight_bank, 1, device)
        else:
            airlight = fixed_airlight
        result = compose_physical_haze(image, density, airlight)
        result = result[:, :, :resized_h, :resized_w]
        result = Resize((original_h, original_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(result)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(result.squeeze(0).clamp(0, 1), save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/train/phys_stage.yaml")
    parser.add_argument("--sd_path", default="weights/v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--controlnet_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="outputs/phys_hazegen")
    parser.add_argument("--prompt", default="dense gray-white foggy haze, thick realistic mist, low visibility, natural color.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--min_size", type=int, default=512)
    parser.add_argument("--depth_dir", default="")
    parser.add_argument("--use_depth_condition", action=BooleanOptionalAction, default=True)
    parser.add_argument("--depth_condition_scale", type=float, default=0.25)
    parser.add_argument("--depth_condition_zero_center", action=BooleanOptionalAction, default=True)
    parser.add_argument("--depth_condition_dropout", type=float, default=0.0)
    parser.add_argument("--invert_depth_condition", action=BooleanOptionalAction, default=False)
    parser.add_argument("--airlight_source", choices=["head", "bank", "fixed"], default="head")
    parser.add_argument("--airlight_bank", default="")
    parser.add_argument("--fixed_airlight", default="0.85,0.88,0.9")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action=BooleanOptionalAction, default=True)
    main(parser.parse_args())
