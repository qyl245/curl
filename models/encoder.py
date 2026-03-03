"""Encoders used by SSL and fusion stages."""
import math
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn


class HybridTimeEncoder(nn.Module):
    """
    通用混合编码器 (CNN + Transformer)
    输入  (B, input_dim, T)
    输出  CLS (B, D)  或  (CLS, seq) 当 return_seq=True
    """

    def __init__(
        self,
        input_dim: int = 1,
        embed_dim: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        # ---- CNN 特征提取 + 下采样 ----
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, embed_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Transformer ----
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

        # ---- CLS token ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- 正弦位置编码 ----
        self.pos_norm = nn.LayerNorm(embed_dim)
        self.register_buffer("_sin_pe", self._make_sin_pe(embed_dim, max_len=1024))

        # ---- CNN 输出长度缓存（替代手动公式）----
        self._seq_len_cache: dict = {}

    # ---- helpers ----
    @staticmethod
    def _make_sin_pe(d: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(1, max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def _get_seq_len(self, T: int) -> int:
        if T not in self._seq_len_cache:
            with torch.no_grad():
                dummy = torch.zeros(1, self.input_dim, T, device=self.cls_token.device)
                self._seq_len_cache[T] = self.cnn(dummy).shape[-1]
        return self._seq_len_cache[T]

    # ---- forward ----
    def forward(
        self, x: torch.Tensor, return_seq: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = x.shape[0]

        h = self.cnn(x)                   # (B, D, S)
        h = h.permute(0, 2, 1)            # (B, S, D)
        S = h.shape[1]

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)    # (B, S+1, D)

        h = h + self._sin_pe[:, :S + 1, :]
        h = self.pos_norm(h)
        h = self.transformer(h)

        cls_emb = h[:, 0]                 # (B, D)
        seq_emb = h[:, 1:]                # (B, S, D)

        return (cls_emb, seq_emb) if return_seq else cls_emb


# ================================================================
# 模态编码器
# ================================================================
class EMGEncoder(nn.Module):
    """
    针对 PCA 后的单通道 EMG。
    """
    def __init__(self, embed_dim=128, nhead=8, num_layers=4):
        super().__init__()
        # 因为做了 PCA，input_dim 固定为 1
        self.backbone = HybridTimeEncoder(input_dim=1, embed_dim=embed_dim, nhead=nhead, num_transformer_layers=num_layers)

    def forward(self, x, return_seq=True):
        return self.backbone(x, return_seq=return_seq)

class IMUEncoder(nn.Module):
    """
    针对 6 轴 IMU。
    """
    def __init__(self, embed_dim=128, nhead=8, num_layers=4):
        super().__init__()
        self.backbone = HybridTimeEncoder(input_dim=6, embed_dim=embed_dim, nhead=nhead, num_transformer_layers=num_layers)

    def forward(self, x, return_seq=True):
        return self.backbone(x, return_seq=return_seq)