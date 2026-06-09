# 导入类型提示相关模块
from typing import Tuple, Set, List, Dict

# 导入PyTorch相关模块
import torch
from torch import nn

# 从本地模块导入相关类
from .controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook


# 定义一个禁用训练模式的函数
def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


# 定义ControlLDM主模型类，继承自nn.Module
class ControlLDM(nn.Module):

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor
    ):
        super().__init__()
        # 初始化UNet模型
        self.unet = ControlledUnetModel(**unet_cfg)
        # 初始化VAE模型
        self.vae = AutoencoderKL(**vae_cfg)
        # 初始化CLIP文本编码器
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        # 初始化ControlNet模型
        self.controlnet = ControlNet(**controlnet_cfg)
        # 设置潜在空间的缩放因子
        self.scale_factor = latent_scale_factor
        # 初始化控制信号缩放系数列表(13个1.0)
        self.control_scales = [1.0] * 13

    # 加载预训练Stable Diffusion模型的权重
    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        # 定义模块名称映射关系
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        # 定义需要加载的模块列表
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()  # 记录已使用的权重键
        missing = set()  # 记录缺失的权重键
        # 遍历所有模块进行权重加载
        for name, module in modules:
            init_sd = {}  # 初始化状态字典
            scratch_sd = module.state_dict()  # 获取模块当前状态字典
            # 遍历模块的所有参数
            for key in scratch_sd:
                # 构建目标权重键
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)  # 记录缺失的键
                    continue
                # 复制权重到初始化字典
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)  # 记录已使用的键
            # 加载权重到模块
            module.load_state_dict(init_sd, strict=False)
        # 计算未使用的权重键
        unused = set(sd.keys()) - used
        # 冻结VAE、CLIP和UNet的参数
        for module in [self.vae, self.clip, self.unet]:
            module.eval()  # 设置为评估模式
            module.train = disabled_train  # 禁用训练模式
            # 关闭所有参数的梯度计算
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    # 从检查点加载ControlNet权重
    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    # 从UNet初始化ControlNet权重
    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        # 获取UNet的状态字典
        unet_sd = self.unet.state_dict()
        # 获取ControlNet的当前状态字典
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}  # 初始化状态字典
        init_with_new_zero = set()  # 记录需要补零初始化的键
        init_with_scratch = set()  # 记录使用随机初始化的键
        # 遍历ControlNet的所有参数
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                # 如果尺寸匹配，直接复制权重
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    # 计算通道差异
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    # 创建零张量补充通道差异
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    # 拼接原始权重和零张量
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                # 保留随机初始化
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        # 加载初始化后的权重
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch

    # VAE编码方法
    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        # 分块编码处理
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                # 使用VAEHook进行分块编码
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                # 量化卷积
                moments = self.vae.quant_conv(h)
                # 创建高斯分布
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        # 采样或取均值
        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    # VAE解码方法
    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        # 分块解码处理
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                # 使用VAEHook进行分块解码
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    # 准备条件输入(图像+文本)
    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),  # 编码文本
            c_img=self.vae_encode(  # 编码图像
                cond_img * 2 - 1,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ),
        )

    # 准备纯文本条件输入
    def prepare_text_condition(
        self,
        txt: List[str],
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),  # 只编码文本
            c_img=None,  # 图像条件为空
        )

    # 前向传播方法
    def forward(self, x_noisy, t, cond):
        c_txt = cond["c_txt"]  # 获取文本条件
        c_img = cond["c_img"]  # 获取图像条件
        # 通过ControlNet获取控制信号
        control = self.controlnet(x=x_noisy, hint=c_img, timesteps=t, context=c_txt)
        # 应用控制信号缩放
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        # 通过UNet预测噪声
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=c_txt,
            control=control,
            only_mid_control=False,
        )
        return eps

    # 转换模型数据类型
    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        self.unet.dtype = dtype  # 设置UNet数据类型
        self.controlnet.dtype = dtype  # 设置ControlNet数据类型
        # 转换UNet各块的数据类型
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # 转换ControlNet各块和零卷积的数据类型
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)

        # 定义GroupNorm32的特殊处理函数(必须使用float32)
        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # 对UNet应用GroupNorm32的特殊处理
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        # 对ControlNet应用GroupNorm32的特殊处理
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)