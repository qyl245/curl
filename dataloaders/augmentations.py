"""
数据增强工具。
"""
import random
from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np
import torch



# ================================================================
# 基础变换（维度安全：对 (C, T) 和 (B, C, T) 均有效）
# ================================================================
def add_gaussian_noise(x: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    return x + torch.randn_like(x) * scale


def time_flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """沿通道轴随机置零。如果是 1 通道(PCA后的EMG)，则跳过或降低概率"""
    C = x.shape[-2]
    if C <= 1: 
        return x # 1个通道掉无可掉
    mask = (torch.rand(C, device=x.device) > p).float()
    shape = [1] * (x.dim() - 2) + [C, 1]
    return x * mask.view(*shape)


def time_mask(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
    """遮蔽一段连续时间"""
    T = x.shape[-1]
    mask_len = max(1, int(T * mask_ratio))
    start = random.randint(0, T - mask_len)
    x = x.clone()
    x[..., start:start + mask_len] = 0
    return x


def time_warp(x: torch.Tensor, lo: float = 0.85, hi: float = 1.15) -> torch.Tensor:
    """时间拉伸 / 压缩后恢复原长"""
    T = x.shape[-1]
    new_T = max(1, int(T * random.uniform(lo, hi)))
    needs_unsqueeze = x.dim() == 2
    if needs_unsqueeze:
        x = x.unsqueeze(0)
    warped = torch.nn.functional.interpolate(
        x.float(), size=new_T, mode="linear", align_corners=False
    )
    if new_T < T:
        warped = torch.cat(
            [warped, torch.zeros(*warped.shape[:-1], T - new_T, device=x.device)],
            dim=-1,
        )
    else:
        warped = warped[..., :T]
    if needs_unsqueeze:
        warped = warped.squeeze(0)
    return warped


def freq_perturb(x: torch.Tensor, scale: float = 0.02) -> torch.Tensor:
    """频域扰动"""
    freq = torch.fft.rfft(x, dim=-1)
    noise = torch.complex(
        torch.randn_like(freq.real) * scale,
        torch.randn_like(freq.imag) * scale,
    )
    return torch.fft.irfft(freq + noise, n=x.shape[-1], dim=-1)


# ================================================================
# 随机组合增广（Phase 1 对比学习用）
# ================================================================
def get_aug_pool(is_emg: bool):
    """根据模态返回增广池"""
    if is_emg:
        return [
            lambda t: add_gaussian_noise(t, scale=random.uniform(0.01, 0.05)),
            lambda t: time_mask(t, mask_ratio=random.uniform(0.05, 0.1)),
            lambda t: time_warp(t, lo=0.9, hi=1.1),
            lambda t: freq_perturb(t, scale=random.uniform(0.01, 0.02)),
        ]
    else: # IMU
        return [
            lambda t: add_gaussian_noise(t, scale=random.uniform(0.02, 0.1)),
            time_flip,
            lambda t: channel_dropout(t, p=random.uniform(0.05, 0.2)),
            lambda t: time_mask(t, mask_ratio=random.uniform(0.05, 0.15)),
            lambda t: time_warp(t, lo=0.85, hi=1.15),
        ]



# ================================================================
# Phase 1: SSL Transform（MAE + 对比 + 顺序预测）
# ================================================================
def create_ssl_transform(
    mask_ratio: float,
    num_views: int,
    num_chunks: int,
    task_type: str = "EMG" # "EMG" 或 "IMU"
):
    """
    适配单模态数据字典。
    输入 sample 包含 "signal": (C, T)
    """
    is_emg = (task_type == "EMG")
    aug_pool = get_aug_pool(is_emg)

    all_perms: List[Tuple[int, ...]] = list(permutations(range(num_chunks)))
    perm_to_index = {p: i for i, p in enumerate(all_perms)}

    def transform(sample: Dict):
        signal = sample["signal"]  # (C, T)
        C, T = signal.shape

        # --- 1. MAE 支路 ---
        # 产生随机遮罩
        mask = torch.rand(T) < mask_ratio
        signal_mae = signal.clone()
        signal_mae[..., mask] = 0

        # --- 2. 对比学习视图 (Contrastive Views) ---
        views = []
        for _ in range(num_views):
            v = signal.clone()
            # 随机选 2 种增广
            chosen = random.sample(aug_pool, k=min(len(aug_pool), 2))
            for fn in chosen:
                v = fn(v)
            views.append(v)
        signal_views = torch.stack(views) # (V, C, T)

        # --- 3. 顺序预测 (Jigsaw/Order Prediction) ---
        chunk_T = T // num_chunks
        if chunk_T > 0:
            chunks = torch.stack(
                [signal[:, i * chunk_T:(i + 1) * chunk_T] for i in range(num_chunks)]
            ) # (K, C, chunk_T)
            perm = random.choice(all_perms)
            signal_order = chunks[list(perm)]
            order_label = perm_to_index[perm]
        else:
            signal_order = signal.unsqueeze(0).repeat(num_chunks, 1, 1)
            order_label = 0

        # 更新字典，只存当前模态需要的
        sample.update({
            "mae_input": signal_mae,
            "mae_label": signal,    # 重构的目标是原信号
            "mae_mask": mask,
            "contrastive_views": signal_views,
            "order_input": signal_order,
            "order_label": torch.tensor(order_label, dtype=torch.long),
        })
        return sample

    return transform


# ================================================================
# Phase 2   
# ================================================================
def scaling(x, sigma=0.1):
    """量级缩放：模拟传感器灵敏度差异或皮肤阻抗变化"""
    if x.dim() == 2:
        factor = torch.tensor(
            np.random.normal(1.0, sigma, (x.shape[0], 1)),
            dtype=x.dtype,
            device=x.device,
        )
    else:
        factor = torch.tensor(
            np.random.normal(1.0, sigma),
            dtype=x.dtype,
            device=x.device,
        )
    return x * factor


def augment_emg(emg: torch.Tensor, profile: str = "light") -> torch.Tensor:
    """
    EMG 增强 — 自动适配单通道(J-tom03)与多通道(MyoGym)
    emg shape: (C, T) 或 (B, C, T)
    """
    x = emg.clone()
    # 获取通道数，通常在倒数第二维
    num_channels = x.shape[-2] 

    if profile == "light":
        # 1. 高斯噪声 (通用)
        if random.random() > 0.65:
            x = add_gaussian_noise(x, 0.03)
        
        # 2. 时间遮掩 (通用)
        if random.random() > 0.7:
            x = time_mask(x, 0.06)
        
        # 3. 通道策略 (差异化处理)
        if num_channels > 1:
            # 多通道：随机丢弃某个通道
            if random.random() > 0.6:
                x = channel_dropout(x, 0.08)
        else:
            # 单通道：进行量级缩放，模拟不同受试者的信号强度差异
            if random.random() > 0.6:
                x = scaling(x, sigma=0.1)
        
        # 4. 时间扭曲 (通用)
        if random.random() > 0.75:
            x = time_warp(x, lo=0.96, hi=1.04)

    elif profile == "default":
        if random.random() > 0.5:
            x = add_gaussian_noise(x, 0.05)
        if random.random() > 0.5:
            x = time_mask(x, 0.1)
        
        # 强增强下的通道处理
        if num_channels > 1:
            if random.random() > 0.3:
                x = channel_dropout(x, 0.15)
        else:
            if random.random() > 0.3:
                x = scaling(x, sigma=0.2) # 更剧烈的缩放
                
        if random.random() > 0.3:
            x = time_warp(x)
        if random.random() > 0.3:
            x = freq_perturb(x)
    else:
        raise ValueError(f"Unsupported EMG augmentation profile: {profile}")
    
    return x

def augment_imu(imu: torch.Tensor) -> torch.Tensor:
    """
    IMU 增强 — 支持 (A, T) 和 (B, A, T)
    修复: 所有操作都通过 shape[-2] 索引轴维度，不再误用 shape[0]
    """
    x = imu.clone()
    if random.random() > 0.5:
        x = add_gaussian_noise(x, 0.05)
    if random.random() > 0.5:
        x = time_mask(x, 0.1)
    if random.random() > 0.3:
        x = channel_dropout(x, 0.15)
    if random.random() > 0.3:
        x = time_warp(x, lo=0.9, hi=1.1)
    if random.random() > 0.3:
        x = freq_perturb(x)
    return x
