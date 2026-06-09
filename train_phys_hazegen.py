from __future__ import annotations

import copy
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from diffbir.model import ControlLDM, Diffusion
from diffbir.model.gaussian_diffusion import extract_into_tensor
from diffbir.sampler import SpacedSampler
from diffbir.utils.common import instantiate_from_config
from tools.hazegen_train_utils import (
    CoATextEncoder,
    encode_clip_image,
    image_to_tensor01,
    init_swanlab,
    list_images,
    load_base_model,
    load_coa_clip,
    random_crop_image,
    read_rgb,
    set_global_seed,
    swanlab_image,
    swanlab_log,
)
from train_hazegen_residual import (
    ClearImageDataset,
    SyntheticHazePairDataset,
    cycle,
    latent_sample_shape,
    prepare_cond,
)
from phys_haze_utils import (
    AirlightHead,
    build_airlight_bank,
    carrier_to_density,
    compose_physical_haze,
    density_to_carrier,
    estimate_density_from_pair,
    load_phys_checkpoint,
    sample_airlight,
    save_phys_checkpoint,
)


DEFAULT_CONFIG = "configs/train/phys_stage.yaml"
DEFAULT_SD_PATH = "weights/v2-1_512-ema-pruned.ckpt"
DEFAULT_PROMPT = "dense gray-white foggy haze, thick realistic mist, low visibility, natural color."


def make_progress(accelerator: Accelerator, total: int, desc: str, enabled: bool) -> tqdm | None:
    """只在交互终端显示动态 tqdm；nohup 日志里改用普通单行日志。"""
    if not (accelerator.is_main_process and enabled and sys.stderr.isatty()):
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)


def update_progress(progress: tqdm | None, metrics: dict[str, float]) -> None:
    """更新 tqdm 时不额外刷新 postfix，避免同一步在日志中出现两次。"""
    if progress is None:
        return
    progress.set_postfix({name: f"{value:.4f}" for name, value in metrics.items()}, refresh=False)
    progress.update(1)


def log_step(accelerator: Accelerator, stage: str, step: int, total: int, every: int, metrics: dict[str, float]) -> None:
    """nohup/日志文件友好的训练状态输出。"""
    if not accelerator.is_main_process:
        return
    if step != 1 and step % every != 0 and step != total:
        return
    metric_text = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    accelerator.print(f"{stage} step {step}/{total} {metric_text}".rstrip())


def swanlab_log_images(swanlab_run, prefix: str, images: dict[str, torch.Tensor], step: int) -> None:
    """把预览图拆成多个 SwanLab panel，而不是拼成一张难读的大图。"""
    if swanlab_run is None:
        return
    data = {}
    for name, tensor in images.items():
        panel_name = name.replace(" ", "_").lower()
        caption = f"{prefix} | {name} | step {step}"
        data[f"{prefix}/{panel_name}"] = swanlab_image(swanlab_run, make_grid(tensor.detach().cpu().clamp(0.0, 1.0), nrow=min(4, tensor.shape[0])), caption)
    swanlab_log(swanlab_run, data, step)


def load_reference_hazy_images(paths: Sequence[Path], crop_size: int, count: int, step: int, device: torch.device) -> torch.Tensor:
    """为 Stage2 预览取真实雾图参考样本。"""
    refs = []
    for offset in range(count):
        path = paths[(step + offset) % len(paths)]
        image = random_crop_image(read_rgb(path), crop_size)
        refs.append(image_to_tensor01(image))
    return torch.stack(refs, dim=0).to(device=device)


def diffusion_p_loss(diffusion: Diffusion, model: ControlLDM, z0: torch.Tensor, cond: dict[str, torch.Tensor]) -> torch.Tensor:
    """标准扩散训练损失。

    这里的 z0 不再是 RGB 雾图 residual 的 latent，而是 density carrier 的 latent:
      density d(x) in [0, 1]
      carrier = repeat(d, 3) * 2 - 1
      z0 = VAE(carrier)
    所以 ControlNet 学的是“雾密度/透射率空间分布”，不是自由 RGB 颜色残差。
    """
    t = torch.randint(0, diffusion.num_timesteps, (z0.shape[0],), device=z0.device).long()
    return diffusion.p_losses(model, z0, t, cond)


def predict_x0_from_output(diffusion: Diffusion, x_noisy: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
    """把模型输出还原成 pred_x0。

    Stage2 的 CLIP 方向约束需要看到当前模型预测出的可视化雾图。
    因此不能只算噪声预测损失，还要从低噪声时间步反推 pred_x0，
    再经 VAE decode 得到 density。
    """
    if diffusion.parameterization == "x0":
        return model_output
    sqrt_alpha = extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, x_noisy.shape)
    sqrt_one_minus = extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
    if diffusion.parameterization == "eps":
        return (x_noisy - sqrt_one_minus * model_output) / sqrt_alpha.clamp_min(1e-8)
    if diffusion.parameterization == "v":
        return sqrt_alpha * x_noisy - sqrt_one_minus * model_output
    raise NotImplementedError(diffusion.parameterization)


def low_noise_density(
    diffusion: Diffusion,
    model: ControlLDM,
    pure_model: ControlLDM,
    z0: torch.Tensor,
    cond: dict[str, torch.Tensor],
    t_max: int,
) -> torch.Tensor:
    """在低噪声时间步估计当前模型的 density 输出。

    只在较小 t 上做可视化/辅助约束，避免高噪声 pred_x0 解码后产生脏纹理，
    也减少 CLIP 对不稳定中间结果的错误牵引。
    """
    b = z0.shape[0]
    max_t = max(1, min(t_max, diffusion.num_timesteps))
    t = torch.randint(0, max_t, (b,), device=z0.device).long()
    noise = torch.randn_like(z0)
    x_noisy = diffusion.q_sample(z0, t, noise)
    model_output = model(x_noisy, t, cond)
    pred_z0 = predict_x0_from_output(diffusion, x_noisy, t, model_output)
    carrier = pure_model.vae_decode(pred_z0)
    return carrier_to_density(carrier)


@torch.no_grad()
def decode_density_latent(pure_model: ControlLDM, z0: torch.Tensor) -> torch.Tensor:
    """把 density latent 解码成可视化/物理合成用的 density map。"""
    return carrier_to_density(pure_model.vae_decode(z0))


@torch.no_grad()
def sample_density_latent(
    model: ControlLDM,
    sampler: SpacedSampler,
    clean: torch.Tensor,
    depth: Optional[torch.Tensor],
    args,
    steps: int,
    cfg_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """用冻结 teacher 采样 pseudo density latent。

    第二阶段没有 paired clean/hazy 标签，所以 teacher 只生成 pseudo density，
    后续颜色由真实雾域 airlight bank 或 airlight head 控制。
    这样避免 teacher 的 RGB 偏色被 student 继承。
    """
    prompt = [args.prompt] * clean.shape[0]
    cond = prepare_cond(model, clean, prompt, depth, args)
    uncond = None
    if cfg_scale != 1.0:
        uncond = {"c_img": torch.zeros_like(cond["c_img"]), "c_txt": copy.deepcopy(cond["c_txt"])}
    return sampler.sample(
        model=model,
        device=device,
        steps=steps,
        x_size=latent_sample_shape(model, cond),
        cond=cond,
        uncond=uncond,
        cfg_scale=cfg_scale,
        progress=False,
    ).detach()


@torch.no_grad()
def load_prompt_direction(clip_module, clip_model, args, device: torch.device) -> torch.Tensor:
    """读取 CLIP prompt 的“加雾方向”。

    优先使用 CoA/CLIP-LIT 训练出的 prompt pair:
      haze_feature - clear_feature
    如果没有可用 checkpoint，就退化为普通文本 prompt 的方向。

    注意这里返回的是方向，不是 haze prototype。也就是说 Stage2 约束的是
    clean -> pred_hazy 的变化是否像“加雾”，而不是强迫 pred_hazy 本身贴近某个雾图中心。
    """
    if args.clip_prompt_ckpt:
        ckpt_path = Path(args.clip_prompt_ckpt)
        if ckpt_path.exists():
            data = torch.load(ckpt_path, map_location="cpu")
            state = {key[7:] if key.startswith("module.") else key: value for key, value in data.items()}
            if "embedding_prompt" in state:
                prompt_embedding = state["embedding_prompt"].to(device).float()
                text_encoder = CoATextEncoder(clip_model).to(device).eval()
                tokenized = torch.cat([clip_module.tokenize(" ".join(["X"] * 16))]).to(device)
                features = text_encoder(prompt_embedding, tokenized)
                features = F.normalize(features.float(), dim=-1)
                if features.shape[0] >= 2:
                    return F.normalize(features[:1] - features[1:2], dim=-1)
                return features[:1]
    haze_tokens = clip_module.tokenize([args.prompt]).to(device)
    clear_tokens = clip_module.tokenize([args.clear_prompt]).to(device)
    haze_feature = F.normalize(clip_model.encode_text(haze_tokens).float(), dim=-1)
    clear_feature = F.normalize(clip_model.encode_text(clear_tokens).float(), dim=-1)
    return F.normalize(haze_feature - clear_feature, dim=-1)


def directional_clip_loss(clip_model, clean: torch.Tensor, pred_hazy: torch.Tensor, text_direction: torch.Tensor) -> torch.Tensor:
    """CLIP 方向损失。

    原始 prototype loss 容易把图像整体往“雾图风格中心”拉，导致颜色/曝光漂移。
    这里改为比较变化方向:
      image_direction = CLIP(pred_hazy) - CLIP(clean)
      text_direction  = CLIP(haze_prompt) - CLIP(clear_prompt)
    """
    clean_feature = encode_clip_image(clip_model, clean)
    pred_feature = encode_clip_image(clip_model, pred_hazy)
    image_direction = F.normalize(pred_feature - clean_feature, dim=-1)
    return 1.0 - (image_direction * text_direction).sum(dim=-1).mean()


def train_stage1(args, cfg, accelerator: Accelerator, device: torch.device, swanlab_run=None) -> Path:
    """第一阶段：从合成 paired 数据学习 clean/depth -> density。

    输入:
      clean: 清晰图 [0, 1]
      hazy:  合成雾图 [-1, 1]
      depth: 可选深度条件 [0, 1]

    训练目标:
      1. 从 paired clean/hazy 估计低频 density target 和 airlight target。
      2. 把 density target 转成 3 通道 carrier，再经 VAE 编码成扩散 z0。
      3. ControlNet 学 z0 去噪；AirlightHead 学低维大气光 A。
    """
    dataset = SyntheticHazePairDataset(
        args.synthetic_clean_dir,
        args.synthetic_hazy_dir,
        args.crop_size,
        args.prompt,
        args.exclude_name_keywords,
        args.synthetic_depth_dir,
        args.use_depth_condition,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # ControlNet 仍从冻结 SD UNet 初始化；训练时只更新 ControlNet 和 AirlightHead。
    model = load_base_model(cfg, args.sd_path, None, device)
    airlight_head = AirlightHead().to(device)
    if args.resume:
        load_phys_checkpoint(args.resume, model, airlight_head)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion).to(device)
    opt = torch.optim.AdamW(
        list(model.controlnet.parameters()) + list(airlight_head.parameters()),
        lr=args.lr,
    )
    model, airlight_head, opt, loader = accelerator.prepare(model, airlight_head, opt, loader)
    exp_dir = Path(args.exp_dir) / "stage1_density"
    ckpt_dir = exp_dir / "checkpoints"
    writer = SummaryWriter(exp_dir) if accelerator.is_main_process else None

    progress = make_progress(accelerator, args.stage1_steps, "Stage1 density", args.progress_bar)
    data_iter = cycle(loader)
    for step in range(1, args.stage1_steps + 1):
        hazy, clean, depth, prompt, _ = next(data_iter)
        hazy = hazy.to(device).float()
        clean = clean.to(device).float()
        depth = depth.to(device).float() if depth.numel() > 0 else None
        pure = accelerator.unwrap_model(model)
        with torch.no_grad():
            # paired 数据只用来估计物理中间量，不直接把 RGB residual 当扩散目标。
            density, airlight_target = estimate_density_from_pair(clean, hazy, blur_kernel=args.density_target_blur)
            z0 = pure.vae_encode(density_to_carrier(density))
            # cond = clean latent + 可选 depth hint + text prompt。
            cond = prepare_cond(pure, clean, prompt, depth, args)

        # diffusion loss 负责学习 density carrier；airlight loss 负责学习低维 A。
        loss_diff = diffusion_p_loss(diffusion, model, z0, cond)
        airlight_pred = airlight_head(clean)
        loss_airlight = F.smooth_l1_loss(airlight_pred, airlight_target)
        loss = loss_diff + args.lambda_airlight * loss_airlight

        opt.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        opt.step()

        if accelerator.is_main_process:
            metrics = {"diff": loss_diff.item(), "air": loss_airlight.item()}
            update_progress(progress, metrics)
            log_step(accelerator, "Stage1 density", step, args.stage1_steps, args.log_every, metrics)
            if writer and step % args.log_every == 0:
                writer.add_scalar("loss/diffusion_density", loss_diff.item(), step)
                writer.add_scalar("loss/airlight", loss_airlight.item(), step)
                writer.add_scalar("loss/total", loss.item(), step)
            if step % args.log_every == 0:
                swanlab_log(
                    swanlab_run,
                    {
                        "stage1/loss_diffusion_density": loss_diff.item(),
                        "stage1/loss_airlight": loss_airlight.item(),
                        "stage1/loss_total": loss.item(),
                    },
                    step,
                )
            if step % args.image_every == 0:
                # 预览顺序: original / generated haze / reference haze / density。
                preview_n = min(4, clean.shape[0])
                original_clean = clean[:preview_n]
                reference_hazy = ((hazy[:preview_n] + 1) * 0.5).clamp(0.0, 1.0)
                generated_hazy = compose_physical_haze(original_clean, density[:preview_n], airlight_pred[:preview_n].detach())
                density_rgb = density[:preview_n].expand(-1, 3, -1, -1)
                grid = make_grid(torch.cat([original_clean, generated_hazy, reference_hazy, density_rgb], dim=0), nrow=preview_n)
                save_path = exp_dir / "preview" / f"{step:07d}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(grid, save_path)
                if writer:
                    writer.add_image("preview/original_generated_reference_density", grid, step)
                swanlab_log_images(
                    swanlab_run,
                    "stage1",
                    {
                        "Original Clean": original_clean,
                        "Generated Haze": generated_hazy,
                        "Reference Haze": reference_hazy,
                        "Estimated Density": density_rgb,
                    },
                    step,
                )
            if step % args.ckpt_every == 0:
                save_phys_checkpoint(
                    ckpt_dir / f"{step:07d}.pt",
                    accelerator.unwrap_model(model),
                    accelerator.unwrap_model(airlight_head),
                    {"stage": "stage1_density"},
                )

    if accelerator.is_main_process:
        if progress is not None:
            progress.close()
        final_path = ckpt_dir / "stage1_final.pt"
        save_phys_checkpoint(final_path, accelerator.unwrap_model(model), accelerator.unwrap_model(airlight_head), {"stage": "stage1_density"})
        if writer:
            writer.close()
    accelerator.wait_for_everyone()
    return ckpt_dir / "stage1_final.pt"


def train_stage2(args, cfg, accelerator: Accelerator, device: torch.device, stage1_ckpt: Optional[Path], swanlab_run=None) -> Path:
    """第二阶段：真实雾域自适应。

    核心思想:
      - teacher 只生成 pseudo density latent，不生成 RGB pseudo haze。
      - 真实雾图目录只用于构建 airlight bank，提供真实雾域的大气光分布。
      - CLIP prompt 分支使用方向约束，判断 clean -> pred_hazy 是否朝雾域变化。
      - student 继续学习 density，同时轻量适配 airlight head。
    """
    init_ckpt = args.stage2_teacher_ckpt or str(stage1_ckpt or "")
    if not init_ckpt:
        raise ValueError("Stage2 needs --stage2_teacher_ckpt or a just-trained Stage1 checkpoint.")

    dataset = ClearImageDataset(
        args.real_clear_dir,
        args.crop_size,
        args.exclude_name_keywords,
        args.real_clear_depth_dir,
        args.use_depth_condition,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # student/teacher 从同一个 Stage1 物理 checkpoint 初始化。
    student = load_base_model(cfg, args.sd_path, None, device)
    teacher = load_base_model(cfg, args.sd_path, None, device)
    student_air = AirlightHead().to(device)
    teacher_air = AirlightHead().to(device)
    load_phys_checkpoint(init_ckpt, student, student_air)
    load_phys_checkpoint(init_ckpt, teacher, teacher_air)
    teacher.eval()
    teacher_air.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    for param in teacher_air.parameters():
        param.requires_grad_(False)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion).to(device)
    teacher_sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)

    # CLIP 只提供“加雾方向”，不再直接把 pred_hazy 拉向雾图 prototype。
    clip_module, clip_model = load_coa_clip(args.clip_model, args.coa_root, device)
    text_direction = load_prompt_direction(clip_module, clip_model, args, device)
    # 从真实雾图中估计 A 分布，Stage2 用它采样真实域大气光。
    airlight_bank = build_airlight_bank(args.real_hazy_dir, args.max_airlight_bank_images, device)
    reference_hazy_paths = list_images(args.real_hazy_dir)
    bank_path = Path(args.exp_dir) / "stage2_real_adapt" / "airlight_bank.pt"
    if accelerator.is_main_process:
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(airlight_bank, bank_path)

    opt = torch.optim.AdamW(
        list(student.controlnet.parameters()) + list(student_air.parameters()),
        lr=args.stage2_lr or args.lr,
    )
    student, student_air, opt, loader = accelerator.prepare(student, student_air, opt, loader)
    exp_dir = Path(args.exp_dir) / "stage2_real_adapt"
    ckpt_dir = exp_dir / "checkpoints"
    writer = SummaryWriter(exp_dir) if accelerator.is_main_process else None

    progress = make_progress(accelerator, args.stage2_steps, "Stage2 real adapt", args.progress_bar)
    data_iter = cycle(loader)
    for step in range(1, args.stage2_steps + 1):
        clean, depth = next(data_iter)
        clean = clean.to(device).float()
        depth = depth.to(device).float() if depth.numel() > 0 else None
        prompt: Sequence[str] = [args.prompt] * clean.shape[0]
        with torch.no_grad():
            # teacher 只负责 pseudo density；颜色不从 teacher 来。
            pseudo_z = sample_density_latent(
                teacher,
                teacher_sampler,
                clean,
                depth,
                args,
                args.teacher_sample_steps,
                args.teacher_cfg_scale,
                device,
            )
            pure_student = accelerator.unwrap_model(student)
            cond = prepare_cond(pure_student, clean, prompt, depth, args)

        # 主监督: student 学 teacher 的 density latent，保持 Stage1 的几何雾结构。
        loss_pseudo = diffusion_p_loss(diffusion, student, pseudo_z, cond)
        aux = clean.shape[0] if args.stage2_aux_batch_size <= 0 else min(args.stage2_aux_batch_size, clean.shape[0])
        clean_aux = clean[:aux]
        pseudo_aux = pseudo_z[:aux]
        # 为 CLIP/可视化构造物理雾图:
        #   teacher pseudo latent -> density，在 no_grad 下解码，避免 VAE decode 挂着大计算图导致 OOM。
        #   student airlight head -> A，CLIP 方向主要约束真实域雾色/大气光。
        density_pred = decode_density_latent(accelerator.unwrap_model(student), pseudo_aux)
        airlight_sample = sample_airlight(airlight_bank, aux, device)
        airlight_pred_aux = student_air(clean_aux)
        pred_hazy = compose_physical_haze(clean_aux, density_pred, airlight_pred_aux)

        loss_clip = torch.tensor(0.0, device=device)
        if args.lambda_clip_direction > 0:
            loss_clip = directional_clip_loss(clip_model, clean_aux, pred_hazy, text_direction)
        loss_air_dist = torch.tensor(0.0, device=device)
        if args.lambda_airlight_stage2 > 0:
            # 让 AirlightHead 的预测靠近真实雾域 A 分布，便于推理时不用随机 bank 也能生成合理颜色。
            loss_air_dist = F.smooth_l1_loss(airlight_pred_aux, airlight_sample)

        loss = loss_pseudo + args.lambda_clip_direction * loss_clip + args.lambda_airlight_stage2 * loss_air_dist
        opt.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        opt.step()

        if accelerator.is_main_process:
            metrics = {"pseudo": loss_pseudo.item(), "clip": loss_clip.item()}
            update_progress(progress, metrics)
            log_step(accelerator, "Stage2 real adapt", step, args.stage2_steps, args.log_every, metrics)
            if writer and step % args.log_every == 0:
                writer.add_scalar("loss/pseudo_density", loss_pseudo.item(), step)
                writer.add_scalar("loss/clip_direction", loss_clip.item(), step)
                writer.add_scalar("loss/airlight_stage2", loss_air_dist.item(), step)
            if step % args.log_every == 0:
                swanlab_log(
                    swanlab_run,
                    {
                        "stage2/loss_pseudo_density": loss_pseudo.item(),
                        "stage2/loss_clip_direction": loss_clip.item(),
                        "stage2/loss_airlight": loss_air_dist.item(),
                        "stage2/loss_total": loss.item(),
                    },
                    step,
                )
            if step % args.image_every == 0:
                # 预览顺序: original / generated haze / real reference haze / density。
                preview_n = min(4, clean_aux.shape[0])
                original_clean = clean_aux[:preview_n]
                generated_hazy = pred_hazy[:preview_n]
                reference_hazy = load_reference_hazy_images(reference_hazy_paths, args.crop_size, preview_n, step, device)
                density_rgb = density_pred[:preview_n].expand(-1, 3, -1, -1)
                grid = make_grid(torch.cat([original_clean, generated_hazy, reference_hazy, density_rgb], dim=0), nrow=preview_n)
                save_path = exp_dir / "preview" / f"{step:07d}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(grid, save_path)
                if writer:
                    writer.add_image("preview/original_generated_reference_density", grid, step)
                swanlab_log_images(
                    swanlab_run,
                    "stage2",
                    {
                        "Original Clean": original_clean,
                        "Generated Haze": generated_hazy,
                        "Reference Haze": reference_hazy,
                        "Pseudo Density": density_rgb,
                    },
                    step,
                )
            if step % args.ckpt_every == 0:
                save_phys_checkpoint(
                    ckpt_dir / f"{step:07d}.pt",
                    accelerator.unwrap_model(student),
                    accelerator.unwrap_model(student_air),
                    {"stage": "stage2_real_adapt", "airlight_bank": str(bank_path)},
                )

    if accelerator.is_main_process:
        if progress is not None:
            progress.close()
        final_path = ckpt_dir / "stage2_final.pt"
        save_phys_checkpoint(final_path, accelerator.unwrap_model(student), accelerator.unwrap_model(student_air), {"stage": "stage2_real_adapt", "airlight_bank": str(bank_path)})
        if writer:
            writer.close()
    accelerator.wait_for_everyone()
    return ckpt_dir / "stage2_final.pt"


def main(args) -> None:
    """统一入口。

    stage=stage1: 只训练 density + airlight 的合成数据阶段。
    stage=stage2: 读取 --stage2_teacher_ckpt 做真实域适配。
    stage=both:   先 Stage1，再用刚保存的 Stage1 checkpoint 继续 Stage2。
    """
    accelerator = Accelerator(split_batches=True)
    set_global_seed(args.seed, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    swanlab_run = init_swanlab(args, cfg, accelerator)
    if accelerator.is_main_process:
        accelerator.print(f"PhysHazeDiffusion stage={args.stage}")
        accelerator.print(f"exp_dir={args.exp_dir}")
    stage1_ckpt = None
    try:
        if args.stage in {"stage1", "both"}:
            stage1_ckpt = train_stage1(args, cfg, accelerator, device, swanlab_run)
        if args.stage in {"stage2", "both"}:
            train_stage2(args, cfg, accelerator, device, stage1_ckpt, swanlab_run)
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    # -------------------------------------------------------------------------
    # 基础路径与阶段选择
    # -------------------------------------------------------------------------
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    parser.add_argument("--sd_path", default=DEFAULT_SD_PATH)
    parser.add_argument("--exp_dir", default="./experiment/phys_hazegen")
    # Stage1 使用 synthetic clean/hazy paired 数据来估计 density/A。
    parser.add_argument("--synthetic_clean_dir", required=True)
    parser.add_argument("--synthetic_hazy_dir", required=True)
    parser.add_argument("--synthetic_depth_dir", default="")
    # Stage2 使用 real clear 作为输入，用 real hazy 提取真实雾域 airlight bank。
    parser.add_argument("--real_clear_dir", required=True)
    parser.add_argument("--real_clear_depth_dir", default="")
    parser.add_argument("--real_hazy_dir", required=True)
    parser.add_argument("--exclude_name_keywords", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--clear_prompt", default="sharp clear clean image, natural color, high visibility")
    # -------------------------------------------------------------------------
    # 深度条件控制
    # depth 只是几何提示，不应该强到主导颜色生成。
    # -------------------------------------------------------------------------
    parser.add_argument("--use_depth_condition", action=BooleanOptionalAction, default=True)
    parser.add_argument("--depth_condition_scale", type=float, default=0.25)
    parser.add_argument("--depth_condition_zero_center", action=BooleanOptionalAction, default=True)
    parser.add_argument("--depth_condition_dropout", type=float, default=0.25)
    parser.add_argument("--invert_depth_condition", action=BooleanOptionalAction, default=False)
    # -------------------------------------------------------------------------
    # 训练规模与优化器
    # -------------------------------------------------------------------------
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--stage1_steps", type=int, default=30000)
    parser.add_argument("--stage2_steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--stage2_lr", type=float, default=0.0)
    # -------------------------------------------------------------------------
    # 物理分解相关权重
    # lambda_airlight: Stage1 学 A。
    # lambda_clip_direction: Stage2 的 CLIP 加雾方向约束。
    # lambda_airlight_stage2: Stage2 让 AirlightHead 贴近真实雾域 A 分布。
    # -------------------------------------------------------------------------
    parser.add_argument("--lambda_airlight", type=float, default=0.1)
    parser.add_argument("--lambda_clip_direction", type=float, default=0.05)
    parser.add_argument("--lambda_airlight_stage2", type=float, default=0.02)
    # density target 先低通，避免把边缘错位/压缩噪声学成雾密度。
    parser.add_argument("--density_target_blur", type=int, default=15)
    # Stage2 teacher 采样 pseudo density 的设置。
    parser.add_argument("--teacher_sample_steps", type=int, default=10)
    parser.add_argument("--teacher_cfg_scale", type=float, default=1.0)
    # 只在低噪声 t 上解码 pred_x0 做 CLIP/可视化，降低脏纹理干扰。
    parser.add_argument("--aux_rgb_t_max", type=int, default=100)
    parser.add_argument("--stage2_aux_batch_size", type=int, default=1)
    # checkpoint: resume 用于 Stage1 续训；stage2_teacher_ckpt 用于只跑 Stage2。
    parser.add_argument("--resume", default="")
    parser.add_argument("--stage2_teacher_ckpt", default="")
    # 日志与保存频率。
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--image_every", type=int, default=1000)
    parser.add_argument("--progress_bar", action=BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=231)
    # CLIP/CoA prompt 方向约束。
    parser.add_argument("--coa_root", default="/data/users/gaoyin/2024_LYB/COA/CoA-main")
    parser.add_argument("--clip_model", default="/data/users/gaoyin/2024_LYB/COA/CoA-main/clip_model/ViT-B-32.pt")
    parser.add_argument("--clip_prompt_ckpt", default="/data/users/gaoyin/2024_LYB/COA/CoA-main/clip_model/haze_prompt.pth")
    # 用多少真实雾图估计 airlight bank。0 以下/过大都不建议，默认 2000 是折中。
    parser.add_argument("--max_airlight_bank_images", type=int, default=2000)
    # SwanLab 默认关闭，避免无意联网/登录；需要时用 --use_swanlab 开启。
    parser.add_argument("--use_swanlab", action=BooleanOptionalAction, default=False)
    parser.add_argument("--swanlab_project", default="PhysHazeDiffusion")
    parser.add_argument("--swanlab_run_name", default="phys_hazegen")
    main(parser.parse_args())
