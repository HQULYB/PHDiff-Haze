import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_path: str | Path) -> list[Path]:
    root = Path(input_path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"input path does not exist: {root}")
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"no images found in: {root}")
    return images


def normalize_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max <= d_min:
        return np.zeros_like(depth, dtype=np.uint8)
    depth = (depth - d_min) / (d_max - d_min)
    return (depth * 255.0).round().clip(0, 255).astype(np.uint8)


def output_path_for(image_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        relative = Path(image_path.stem + ".png")
    else:
        relative = image_path.relative_to(input_root).with_suffix(".png")
    return output_dir / relative


class HFDepthAnything:
    def __init__(self, model_path: str, device: torch.device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True).to(device).eval()

    @torch.no_grad()
    def infer(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth = F.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        return depth.squeeze().detach().cpu().numpy()


class OfficialDepthAnything:
    def __init__(self, repo: str, checkpoint: str, encoder: str, input_size: int, device: torch.device):
        sys.path.insert(0, str(Path(repo).resolve()))
        from depth_anything_v2.dpt import DepthAnythingV2

        configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        self.device = device
        self.input_size = input_size
        self.model = DepthAnythingV2(**configs[encoder])
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def infer(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        return self.model.infer_image(image, self.input_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or folder.")
    parser.add_argument("--output_dir", required=True, help="Folder to save grayscale depth png files.")
    parser.add_argument("--backend", choices=["hf", "official"], default="hf")
    parser.add_argument("--hf_model", default="/data/users/gaoyin/datasets/Depth-Anything-V2-Small-hf")
    parser.add_argument("--official_repo", default="/data/users/gaoyin/2024_LYB/Depth-Anything-V2")
    parser.add_argument("--official_ckpt", default="/data/users/gaoyin/2024_CKB/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitb")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_root = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "hf":
        estimator = HFDepthAnything(args.hf_model, device)
    else:
        estimator = OfficialDepthAnything(args.official_repo, args.official_ckpt, args.encoder, args.input_size, device)

    image_paths = list_images(input_root)
    for image_path in tqdm(image_paths, desc="Precompute depth"):
        save_path = output_path_for(image_path, input_root, output_dir)
        if args.skip_existing and save_path.exists():
            continue
        depth = estimator.infer(image_path)
        depth_u8 = normalize_to_uint8(depth)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), depth_u8)


if __name__ == "__main__":
    main()
