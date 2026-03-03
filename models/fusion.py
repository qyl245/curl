import torch
import torch.nn as nn
from typing import Dict


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetEncoder(nn.Module):
    """
    兼容多通道输入时的集合编码器。
    输入: (B, C, D)
    输出: global (B, D), seq (B, C, D)
    """

    def __init__(self, embed_dim: int, nhead: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        seq = self.encoder(x)
        glob = self.norm(seq.mean(dim=1))
        return glob, seq


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int, nhead: int):
        super().__init__()
        self.cross_emg2imu = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.cross_imu2emg = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)

        self.norm_emg_seq = nn.LayerNorm(embed_dim)
        self.norm_imu_seq = nn.LayerNorm(embed_dim)
        self.norm_emg_glob = nn.LayerNorm(embed_dim)
        self.norm_imu_glob = nn.LayerNorm(embed_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, emg_ctx, imu_ctx, emg_global, imu_global):
        emg_cross, _ = self.cross_emg2imu(query=emg_ctx, key=imu_ctx, value=imu_ctx)
        imu_cross, _ = self.cross_imu2emg(query=imu_ctx, key=emg_ctx, value=emg_ctx)

        emg_enh = self.norm_emg_seq(emg_ctx + emg_cross)
        imu_enh = self.norm_imu_seq(imu_ctx + imu_cross)

        emg_enh_g = emg_enh.mean(dim=1)
        imu_enh_g = imu_enh.mean(dim=1)

        emg_feat = self.norm_emg_glob(emg_global + emg_enh_g)
        imu_feat = self.norm_imu_glob(imu_global + imu_enh_g)

        fused = self.fusion_mlp(torch.cat([emg_feat, imu_feat], dim=-1))
        return fused, emg_feat, imu_feat


class MultiModalModel(nn.Module):
    def __init__(self, emg_encoder, imu_encoder, model_cfg, num_classes: int):
        super().__init__()
        self.emg_encoder = emg_encoder
        self.imu_encoder = imu_encoder

        d_model = model_cfg["fusion"]["embed_dim"]
        fcfg = model_cfg["fusion"]

        self.use_single_channel = model_cfg["emg"]["num_channels"] == 1
        if not self.use_single_channel:
            self.emg_set = SetEncoder(d_model, nhead=fcfg.get("set_heads", 4))

        dropout = fcfg.get("dropout", 0.0)
        self.fusion = CrossModalFusion(d_model, fcfg["fusion_heads"])
        self.modality_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self.proj_cross = ProjectionHead(d_model, d_model)
        self.proj_emg = ProjectionHead(d_model, d_model)
        self.proj_imu = ProjectionHead(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _encode_emg(self, emg: torch.Tensor):
        if self.use_single_channel:
            return self.emg_encoder(emg, return_seq=True)
        emg_tokens = self.emg_encoder(emg, return_seq=False)
        return self.emg_set(emg_tokens)

    def _encode_imu(self, imu: torch.Tensor):
        return self.imu_encoder(imu, return_seq=True)

    def forward(self, emg: torch.Tensor, imu: torch.Tensor) -> Dict[str, torch.Tensor]:
        emg_global, emg_ctx = self._encode_emg(emg)
        imu_global, imu_ctx = self._encode_imu(imu)

        fused, emg_feat, imu_feat = self.fusion(emg_ctx, imu_ctx, emg_global, imu_global)
        gate_logits = self.modality_gate(torch.cat([emg_feat, imu_feat], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        gated = gate[:, :1] * emg_feat + gate[:, 1:] * imu_feat
        fused_final = 0.5 * (fused + gated)

        return {
            "logits": self.classifier(fused_final),
            "fused_proj": self.proj_cross(fused_final),
            "emg_proj": self.proj_emg(emg_feat),
            "imu_proj": self.proj_imu(imu_feat),
            "modality_gate": gate,
            "fused_embedding": fused_final,
        }