"""Phase 1 self-supervised model: MAE + contrastive + order prediction."""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import info_nce_loss


class MAEDecoder(nn.Module):
    """从序列嵌入重建原始信号"""

    def __init__(self, output_dim: int = 1, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(64, output_dim, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, seq_emb: torch.Tensor) -> torch.Tensor:
        """seq_emb: (B, S, D) → (B, output_dim, T_recon)"""
        return self.decoder(seq_emb.permute(0, 2, 1))


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int = 128, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, proj_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class OrderPredictionHead(nn.Module):
    """多分类排列预测（num_chunks! 类）"""

    def __init__(self, embed_dim: int = 128, num_chunks: int = 4, dropout: float = 0.1):
        super().__init__()
        num_perms = math.factorial(num_chunks)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, batch_first=True,
            dropout=dropout, dim_feedforward=embed_dim * 2,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_perms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, D) → logits (B, num_perms)"""
        return self.classifier(self.transformer(x).mean(dim=1))


# ================================================================
# SSLModel
# ================================================================
class SSLModel(nn.Module):
    """
    Phase 1 自监督模型（EMG 或 IMU 单模态）。
    encoder 是 HybridTimeEncoder。
    """

    def __init__(self, encoder: nn.Module, config: dict, modality: str = "emg"):
        super().__init__()
        self.encoder = encoder
        self.modality = modality

        mcfg = config["model"][modality]
        tcfg = config["train"]["phase1"]
        D = mcfg["embed_dim"]
        self.signal_dim = 1 if self.modality == "emg" else 6
        self.num_channels = mcfg.get("num_channels", 1)
        self.temperature = tcfg["temperature"]

        self.mae_decoder = MAEDecoder(
            output_dim=self.signal_dim, embed_dim=D, dropout=mcfg["dropout"]
        )
        self.proj_head = ProjectionHead(D, D, mcfg["dropout"])
        self.order_head = OrderPredictionHead(D, tcfg["num_chunks"], mcfg["dropout"])

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        """
        使用 augmentations.py 中定义的通用键名
        """
        losses: Dict[str, torch.Tensor] = {}

        # 1. MAE Loss
        if "mae_input" in batch:
            losses["mae"] = self._mae_loss(
                batch["mae_input"], batch["mae_label"], batch["mae_mask"]
            )

        # 2. Contrastive Loss
        if "contrastive_views" in batch:
            losses["contrastive"] = self._contrastive_loss(batch["contrastive_views"])

        # 3. Jigsaw Loss
        if "order_input" in batch:
            losses["order"] = self._order_loss(batch["order_input"], batch["order_label"])

        return losses

    def _mae_loss(self, masked, original, mask):
        """
        masked: (B, C, T)
        original: (B, C, T)
        mask: (B, T) - 布尔矩阵，True表示被遮盖
        """
        # Encoder 提取特征 (B, S, D)
        _, seq = self.encoder(masked, return_seq=True)
        
        # Decoder 重构信号 (B, C, T_recon)
        recon = self.mae_decoder(seq)

        # 对齐长度
        T = original.shape[-1]
        recon = recon[..., :T]

        # 仅在被遮盖的位置计算时间域 MSE
        # mask: (B, T) -> (B, 1, T) -> (B, C, T)
        mask_exp = mask.unsqueeze(1).expand_as(original)
        
        if mask_exp.any():
            loss_time = F.mse_loss(recon[mask_exp], original[mask_exp])
        else:
            loss_time = F.mse_loss(recon, original) # 防止异常

        # 频域 Loss (全局，保证信号整体频谱特性)
        loss_freq = F.mse_loss(
            torch.fft.rfft(recon.float(), dim=-1).abs(),
            torch.fft.rfft(original.float(), dim=-1).abs(),
        )
        return loss_time + 0.5 * loss_freq

    def _contrastive_loss(self, views):
        """
        views: (B, V, C, T)
        """
        B, V, C, T = views.shape
        # 将 Batch 和 View 合并送入 Encoder
        flat_views = views.view(B * V, C, T)
        
        # 提取 CLS token (B*V, D)
        cls_emb = self.encoder(flat_views, return_seq=False)
        
        # 映射到投影空间
        proj = self.proj_head(cls_emb).view(B, V, -1) # (B, V, D)
        
        # 调用 info_nce_loss (需要确保 utils.losses 中有此函数)
        return info_nce_loss(proj, self.temperature)

    def _order_loss(self, order_seq, label):
        """
        order_seq: (B, K, C, chunk_T)
        """
        B, K, C, cT = order_seq.shape
        # 合并送入 Encoder
        flat_chunks = order_seq.view(B * K, C, cT)
        
        # 提取各块的特征 (B*K, D)
        cls_chunks = self.encoder(flat_chunks, return_seq=False)
        
        # 映射回 (B, K, D)
        cls_chunks = cls_chunks.view(B, K, -1)
        
        # 预测排列类别
        logits = self.order_head(cls_chunks)
        return F.cross_entropy(logits, label)