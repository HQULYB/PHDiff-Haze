import importlib
import random
import sys
import types
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from diffbir.model import ControlLDM
from diffbir.utils.common import instantiate_from_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def set_global_seed(seed: int, device_specific: bool = True) -> None:
    """统一设置 Python / NumPy / PyTorch / Accelerate 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed, device_specific=device_specific)


def init_swanlab(args, cfg, accelerator):
    """在主进程初始化 SwanLab。"""
    if not args.use_swanlab or not accelerator.is_main_process:
        return None
    try:
        import swanlab

        swanlab.login(api_key="cyYmfrQdf2uJpux5XA9cl", save=False)
    except ImportError as exc:
        raise ImportError(
            "已开启 SwanLab 日志，但当前环境没有安装 swanlab。"
            "请先执行 `pip install swanlab`，或加 `--no-use_swanlab` 关闭云端日志。"
        ) from exc

    config = vars(args).copy()
    config["model_config"] = OmegaConf.to_container(cfg, resolve=True)
    swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_run_name,
        config=config,
    )
    return swanlab


def swanlab_log(swanlab_run, data: dict, step: int) -> None:
    """封装 SwanLab log，便于在未启用时无操作。"""
    if swanlab_run is not None:
        swanlab_run.log(data, step=step)


def swanlab_image(swanlab_run, tensor: torch.Tensor, caption: str):
    """把 CHW 图像网格 Tensor 转为 SwanLab 图片对象。"""
    if swanlab_run is None:
        return None
    image = to_pil_image(tensor.detach().cpu().clamp(0.0, 1.0))
    return swanlab_run.Image(image, caption=caption)


def list_images(folder: str | Path) -> list[Path]:
    """递归收集目录下的图像文件，支持常见图片后缀。"""
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"image folder does not exist: {root}")
    if root.is_file():
        return [root]
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"no images found in: {root}")
    return images


def read_rgb(path: Path) -> np.ndarray:
    """用 OpenCV 读取图像，并从 BGR 转为训练中使用的 RGB。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_shorter_than(image: np.ndarray, crop_size: int) -> np.ndarray:
    """如果图像短边小于 crop_size，先等比例放大，保证后续能裁剪。"""
    h, w = image.shape[:2]
    short = min(h, w)
    if short >= crop_size:
        return image
    scale = crop_size / max(short, 1)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)


def paired_random_crop(clean: np.ndarray, hazy: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """对 clean/hazy 配对图做同步随机裁剪和翻转，保持像素级对应关系。"""
    clean = resize_shorter_than(clean, crop_size)
    hazy = resize_shorter_than(hazy, crop_size)
    h = min(clean.shape[0], hazy.shape[0])
    w = min(clean.shape[1], hazy.shape[1])
    clean = clean[:h, :w]
    hazy = hazy[:h, :w]
    if h > crop_size and w > crop_size:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        clean = clean[top : top + crop_size, left : left + crop_size]
        hazy = hazy[top : top + crop_size, left : left + crop_size]
    else:
        clean = cv2.resize(clean, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        hazy = cv2.resize(hazy, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    if random.random() < 0.5:
        clean = np.ascontiguousarray(clean[:, ::-1])
        hazy = np.ascontiguousarray(hazy[:, ::-1])
    return clean, hazy


def random_crop_image(image: np.ndarray, crop_size: int) -> np.ndarray:
    """对单张真实清晰图做随机裁剪和翻转。"""
    image = resize_shorter_than(image, crop_size)
    h, w = image.shape[:2]
    if h > crop_size and w > crop_size:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        image = image[top : top + crop_size, left : left + crop_size]
    else:
        image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
    return image


def image_to_tensor01(image: np.ndarray) -> torch.Tensor:
    """把 RGB uint8 图像转成 [0, 1] 范围的 CHW Tensor。"""
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().clamp(0.0, 1.0)


def pair_key(path: Path) -> str:
    """配对名辅助函数。比如 0001_0.8_0.12.png 会尝试匹配 0001.jpg。"""
    return path.stem.split("_")[0]


def load_coa_clip(clip_model: str, clip_root: str, device: torch.device):
    """加载 CoA 项目中的 CLIP 实现和权重。"""
    coa_root = Path(clip_root).resolve()
    clip_dir = coa_root / "CLIP"
    package_name = "coa_clip_runtime"
    package = types.ModuleType(package_name)
    package.__path__ = [str(clip_dir)]
    sys.modules.setdefault(package_name, package)
    clip = importlib.import_module(f"{package_name}.clip")
    model, _ = clip.load(clip_model, device=device, jit=False, download_root=str(coa_root / "clip_model"))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return clip, model


def normalize_clip_image(images: torch.Tensor) -> torch.Tensor:
    """把 [0, 1] RGB 图像缩放并归一化为 CLIP 图像编码器输入。"""
    images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    mean = torch.tensor(CLIP_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def global_clip_feature(raw_feature):
    """兼容 CoA 的 CLIP 输出，返回 L2 归一化的全局特征。"""
    if isinstance(raw_feature, (tuple, list)):
        raw_feature = raw_feature[0]
    if raw_feature.ndim == 3:
        raw_feature = raw_feature[:, 0, :]
    return F.normalize(raw_feature.float(), dim=-1)


def encode_clip_image(clip_model, images: torch.Tensor) -> torch.Tensor:
    """提取并 L2 归一化 CLIP 图像全局特征。"""
    clip_input = normalize_clip_image(images)
    return global_clip_feature(clip_model.encode_image(clip_input))


class CoATextEncoder(nn.Module):
    """复用 CoA/OpenAI CLIP 的文本编码器，把训练好的 prompt embedding 编码成文本特征。"""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        eot_indices = tokenized_prompts.argmax(dim=-1).to(x.device)
        if eot_indices.numel() == 1 and x.shape[0] > 1:
            eot_indices = eot_indices.repeat(x.shape[0])
        elif eot_indices.numel() != x.shape[0]:
            raise ValueError(
                f"tokenized_prompts batch ({eot_indices.numel()}) does not match prompt batch ({x.shape[0]})"
            )

        return x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection


@torch.no_grad()
def load_trained_haze_prompt_feature(
    clip_module,
    clip_model,
    prompt_ckpt: Optional[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """加载训练好的 CLIP haze prompt 权重，并返回 haze 文本特征。"""
    if not prompt_ckpt:
        return None

    ckpt_path = Path(prompt_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CLIP prompt checkpoint does not exist: {ckpt_path}")

    data = torch.load(ckpt_path, map_location="cpu")
    state = {}
    for key, value in data.items():
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value
    if "embedding_prompt" not in state:
        raise KeyError(f"`embedding_prompt` not found in CLIP prompt checkpoint: {ckpt_path}")

    embedding_prompt = state["embedding_prompt"].to(device).float()
    text_encoder = CoATextEncoder(clip_model).to(device).eval()
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    tokenized_prompts = torch.cat([clip_module.tokenize(" ".join(["X"] * 16))]).to(device)
    text_features = text_encoder(embedding_prompt, tokenized_prompts)
    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features[:1]


@torch.no_grad()
def build_haze_prototype(
    clip_module,
    clip_model,
    real_hazy_dir: Optional[str],
    prompt: str,
    batch_size: int,
    device: torch.device,
    max_images: int,
) -> torch.Tensor:
    """构建真实雾域 CLIP 原型。优先使用真实雾图目录，否则退化到文本 prompt。"""
    if real_hazy_dir:
        paths = list_images(real_hazy_dir)
        if max_images > 0:
            paths = paths[:max_images]
        features = []
        for start in tqdm(range(0, len(paths), batch_size), desc="CLIP real haze prototype"):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                image = read_rgb(path)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(image_to_tensor01(image))
            images_t = torch.stack(images).to(device)
            features.append(encode_clip_image(clip_model, images_t))
        prototype = torch.cat(features, dim=0).mean(dim=0, keepdim=True)
        return F.normalize(prototype, dim=-1)

    tokens = clip_module.tokenize([prompt]).to(device)
    text_feature = clip_model.encode_text(tokens)
    return F.normalize(text_feature.float(), dim=-1)


def update_ema(model: ControlLDM, ema_model: ControlLDM, decay: float) -> None:
    """只对 ControlNet 参数做 EMA 平滑。"""
    model_state = model.controlnet.state_dict()
    ema_state = ema_model.controlnet.state_dict()
    with torch.no_grad():
        for key, value in model_state.items():
            ema_value = ema_state[key]
            ema_value.mul_(decay).add_(value.detach().to(device=ema_value.device), alpha=1.0 - decay)


def save_controlnet(path: Path, model: ControlLDM) -> None:
    """只保存 ControlNet 权重，保持和原项目 stage1.pt 格式一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.controlnet.state_dict(), path)


def load_base_model(cfg, sd_path: str, controlnet_path: Optional[str], device: torch.device) -> ControlLDM:
    """构建 ControlLDM，加载 Stable Diffusion 权重，并初始化/恢复 ControlNet。"""
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(sd_path, map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd)
    if controlnet_path:
        cldm.load_controlnet_from_ckpt(torch.load(controlnet_path, map_location="cpu"))
    else:
        cldm.load_controlnet_from_unet()
    return cldm.to(device)
