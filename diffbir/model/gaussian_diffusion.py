# 从functools模块导入partial函数
from functools import partial
# 从typing模块导入Tuple类型提示
from typing import Tuple

# 导入torch库
import torch
# 从torch中导入nn模块
from torch import nn
# 导入numpy库并简写为np
import numpy as np


# 定义创建beta调度表的函数
def make_beta_schedule(
    schedule: str, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    # 如果是线性调度
    if schedule == "linear":
        # 创建线性间隔的beta值
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )

    # 如果是余弦调度
    elif schedule == "cosine":
        # 创建时间步长数组
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        # 计算alpha值
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        # 计算beta值
        betas = 1 - alphas[1:] / alphas[:-1]
        # 限制beta值范围
        betas = np.clip(betas, a_min=0, a_max=0.999)

    # 如果是平方根线性调度
    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    # 如果是平方根调度
    elif schedule == "sqrt":
        betas = (
            np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
        )
    else:
        # 未知调度类型报错
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


# 定义从张量中提取特定时间步数据的函数
def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]
) -> torch.Tensor:
    # 获取batch大小
    b, *_ = t.shape
    # 从a中按索引t提取数据
    out = a.gather(-1, t)
    # 重塑输出张量形状
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# 从指定URL复制的代码，实现零终端信噪比强制
# 原始论文: https://arxiv.org/abs/2305.08891
def enforce_zero_terminal_snr(betas: np.ndarray) -> np.ndarray:
    # 将numpy数组转换为torch张量
    betas = torch.from_numpy(betas)
    # 将beta转换为alpha
    alphas = 1 - betas
    # 计算alpha的累积乘积
    alphas_bar = alphas.cumprod(0)
    # 计算alpha_bar的平方根
    alphas_bar_sqrt = alphas_bar.sqrt()

    # 存储原始值
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # 平移使最后一个时间步为零
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # 缩放使第一个时间步回到原始值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # 将alphas_bar_sqrt转换回betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    # 返回numpy数组
    return betas.numpy()


# 定义扩散模型类，继承自nn.Module
class Diffusion(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",
        zero_snr=False
    ):
        # 调用父类构造函数
        super().__init__()
        # 设置时间步数
        self.num_timesteps = timesteps
        # 设置beta调度类型
        self.beta_schedule = beta_schedule
        # 设置线性调度的起始值
        self.linear_start = linear_start
        # 设置线性调度的结束值
        self.linear_end = linear_end
        # 设置余弦调度的参数
        self.cosine_s = cosine_s
        # 检查参数化类型是否有效
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], "currently only supporting 'eps' and 'x0' and 'v'"
        # 设置参数化类型
        self.parameterization = parameterization
        # 设置是否使用零信噪比
        self.zero_snr = zero_snr
        # 设置损失类型
        self.loss_type = loss_type

        # 创建beta调度表
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        # 如果需要零终端信噪比
        if zero_snr:
            betas = enforce_zero_terminal_snr(betas)
        # 计算alpha值
        alphas = 1.0 - betas
        # 计算alpha的累积乘积
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # 计算alpha累积乘积的平方根
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        # 计算1减去alpha累积乘积的平方根
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        # 存储beta值
        self.betas = betas
        # 注册sqrt_alphas_cumprod为缓冲区
        self.register("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        # 注册sqrt_one_minus_alphas_cumprod为缓冲区
        self.register("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    # 定义注册缓冲区的辅助方法
    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    # 定义前向扩散过程
    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    # 定义计算v预测的方法
    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    # 定义计算损失的方法
    def get_loss(self, pred, target, mean=True):
        # 如果是L1损失
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        # 如果是L2损失
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            # 未知损失类型报错
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # 定义计算模型损失的方法
    def p_losses(self, model, x_start, t, cond):
        # 生成随机噪声
        noise = torch.randn_like(x_start)
        # 对输入进行前向扩散
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 获取模型输出
        model_output = model(x_noisy, t, cond)

        # 根据参数化类型设置目标值
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # 计算简单损失
        loss_simple = self.get_loss(model_output, target, mean=False).mean()
        return loss_simple