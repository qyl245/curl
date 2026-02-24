"""
Phase 1 SSL 训练器
  - AMP 混合精度（统一使用新版 torch.amp API）
  - Warmup + Cosine LR（自实现，去掉 transformers 依赖）
  - 梯度裁剪
  - TensorBoard 日志
  - Early Stopping（基于验证损失）
  - 编码器权重保存
"""
import os
import math
from collections import defaultdict
from typing import Dict

import torch
import torch.nn.utils
from torch.optim import AdamW
from tqdm import tqdm

from utils.logging_utils import setup_logger

logger = setup_logger()

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            logger.warning("tensorboard 未安装，日志将仅输出到控制台。")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None


# ================================================================
# Warmup + Cosine Scheduler（自实现，无需 transformers）
# ================================================================
def _create_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """返回 LambdaLR：线性 warmup → cosine 衰减"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ================================================================
# SSLTrainer
# ================================================================
class SSLTrainer:
    def __init__(self, model, train_loader, val_loader, config: dict, modality: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.modality = modality

        tcfg = config["train"]["phase1"]
        self.loss_weights: Dict[str, float] = tcfg["loss_weights"]
        self.num_epochs = tcfg["num_epochs"]
        self.patience = tcfg.get("patience", 15)

        # 设备
        self.device = config["train"]["device"]
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            self.device = "cpu"
        self.model.to(self.device)

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )

        # Scheduler
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = len(train_loader) * tcfg.get("warmup_epochs", 5)
        self.scheduler = _create_cosine_scheduler(
            self.optimizer, warmup_steps, total_steps
        )

        # AMP（新版 API）
        self.use_amp = self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 输出
        out_dir = config["train"]["output_dir"]
        self.model_dir = os.path.join(out_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(out_dir, "tensorboard", f"ssl_{modality}"))

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        logger.info(
            f"SSLTrainer({modality}) — device={self.device}, "
            f"epochs={self.num_epochs}, loss_weights={self.loss_weights}"
        )

    # ---- 工具 ----
    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _weighted_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        for k, v in losses.items():
            w = self.loss_weights.get(k, 1.0)
            total = total + v * w
        return total

    # ---- 训练 ----
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_losses = defaultdict(float)
        total, n = 0.0, 0

        pbar = tqdm(self.train_loader, desc=f"SSL-{self.modality} E{epoch+1} [Train]")
        for batch in pbar:
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                losses = self.model(**batch)
                loss = self._weighted_loss(losses)

            if not torch.isfinite(loss):
                logger.warning(
                    f"[{self.modality}] 跳过非有限训练损失: {loss.detach().float().item()}"
                )
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total += loss.item()
            for k, v in losses.items():
                epoch_losses[k] += v.item()            
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total / max(n, 1)
        self.writer.add_scalar("Loss/Train", avg, epoch)
        for k, v in epoch_losses.items():
            self.writer.add_scalar(f"Loss/Train_{k}", v / n, epoch)
        self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)
        return avg

    # ---- 验证 ----
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        epoch_losses = defaultdict(float)
        total, n = 0.0, 0

        for batch in tqdm(self.val_loader, desc=f"SSL-{self.modality} E{epoch+1} [Val]"):
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                losses = self.model(**batch)
                loss = self._weighted_loss(losses)
            if not torch.isfinite(loss):
                logger.warning(
                    f"[{self.modality}] 跳过非有限验证损失: {loss.detach().float().item()}"
                )
                continue
            total += loss.item()

            for k, v in losses.items():
                epoch_losses[k] += v.item()    

            n += 1

        avg = total / max(n, 1)
        self.writer.add_scalar("Loss/Val", avg, epoch)
        for k, v in epoch_losses.items():
            self.writer.add_scalar(f"Loss/Val_{k}", v / n, epoch)        
        return avg

    # ---- 主循环 ----
    def train(self):
        logger.info(f"开始 Phase 1 SSL 训练 ({self.modality})...")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            logger.info(
                f"[{self.modality}] Epoch {epoch+1}/{self.num_epochs} — "
                f"Train: {train_loss:.4f}, Val: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_encoder(tag="best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(patience={self.patience})"
                    )
                    break

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        self.save_encoder(tag="final")
        self.writer.close()
        logger.info(f"Phase 1 SSL ({self.modality}) 训练完成!")

    # ---- 保存 ----
    def save_checkpoint(self, epoch: int):
        path = os.path.join(self.model_dir, f"ssl_{self.modality}_ep{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def save_encoder(self, tag: str = "best"):
        path = os.path.join(self.model_dir, f"{self.modality}_encoder_{tag}.pt")
        torch.save(self.model.encoder.state_dict(), path)
        logger.info(f"Encoder saved: {path}")
