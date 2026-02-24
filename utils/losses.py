"""
共享损失函数与张量工具
"""
from typing import Optional

import torch
import torch.nn.functional as F


# ================================================================
# 通用工具
# ================================================================
def masked_mean_pooling(
    features: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    """
    带掩码的平均池化。
    features: (B, C, ...)  mask: (B, C) True=有效  dim: 聚合维度
    """
    if mask is None:
        return features.mean(dim=dim)
    expanded_shape = list(mask.shape) + [1] * (features.ndim - mask.ndim)
    mask_expanded = mask.view(expanded_shape).to(features.dtype)
    sum_features = (features * mask_expanded).sum(dim=dim)
    count = mask_expanded.sum(dim=dim).clamp(min=1e-6)
    return sum_features / count


# ================================================================
# InfoNCE
# ================================================================
def info_nce_loss(
    features: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE 对比损失。
    features: (B, V, D) — V 个视图的嵌入
    """
    B, V, D = features.shape
    device = features.device
    if V < 2:
        raise ValueError(f"Need >= 2 views, got {V}")

    features = F.normalize(features, dim=2)
    flat = features.view(B * V, D)
    sim = torch.matmul(flat, flat.T) / temperature

    # 正样本：同一样本的其他视图
    labels = torch.arange(B * V, device=device)
    for i in range(B):
        for v in range(V):
            labels[i * V + v] = i * V + ((v + 1) % V)

    # 屏蔽自身
    eye = torch.eye(B * V, device=device).bool()
    sim.masked_fill_(eye, -float("inf"))

    return F.cross_entropy(sim, labels)


def info_nce_symmetric(
    a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    对称 InfoNCE (Phase 2 跨模态 / 模态内对比)。
    a, b: (B, D)
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = torch.matmul(a, b.T) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
