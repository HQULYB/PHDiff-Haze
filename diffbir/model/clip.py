# 导入类型提示模块
from typing import List
# 导入PyTorch核心库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入梯度检查点功能
from torch.utils.checkpoint import checkpoint
# 从本地模块导入CLIP模型和分词器
from .open_clip import CLIP, tokenize


class FrozenOpenCLIPEmbedder(nn.Module):
    """
    使用OpenCLIP的文本编码器（冻结权重版）
    """
    # 可选的文本特征提取层
    LAYERS = [
        # "pooled",  # 池化层（已注释）
        "last",  # 最后一层
        "penultimate"  # 倒数第二层
    ]

    def __init__(self, embed_dim, vision_cfg, text_cfg, layer="last"):
        super().__init__()  # 调用父类初始化
        assert layer in self.LAYERS  # 验证层选择是否合法

        # 初始化CLIP模型（转配置为字典格式）
        model = CLIP(embed_dim, dict(vision_cfg), dict(text_cfg))
        del model.visual  # 删除视觉模块（仅保留文本编码）
        self.model = model  # 保存模型主体

        # 设置目标特征层
        self.layer = layer
        # 根据选择设置层索引
        if self.layer == "last":
            self.layer_idx = 0  # 最后一层
        elif self.layer == "penultimate":
            self.layer_idx = 1  # 倒数第二层
        else:
            raise NotImplementedError()  # 其他层未实现

    def forward(self, tokens):
        # 前向传播入口
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        # 文本编码主流程
        x = self.model.token_embedding(text)  # 获取词嵌入 [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding  # 添加位置编码
        x = x.permute(1, 0, 2)  # 调整维度顺序 NLD -> LND
        # 执行Transformer编码
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # 恢复维度顺序 LND -> NLD
        x = self.model.ln_final(x)  # 最终层归一化
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        # Transformer层迭代处理
        for i, r in enumerate(self.model.transformer.resblocks):
            # 到达目标层时停止
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            # 根据条件使用梯度检查点
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)  # 节省显存的模式
            else:
                x = r(x, attn_mask=attn_mask)  # 常规前向传播
        return x

    def encode(self, text: List[str]) -> torch.Tensor:
        # 对外文本编码接口
        tokens = tokenize(text)  # 文本分词
        # 自动匹配模型设备
        tokens = tokens.to(next(self.model.parameters()).device)
        return self(tokens)  # 执行完整编码流程