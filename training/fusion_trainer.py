"""
Phase 2 跨模态融合训练器。
"""
import json
import math
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.optim import AdamW
from tqdm import tqdm

from dataloaders.augmentations import augment_emg, augment_imu
from utils.losses import info_nce_symmetric
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


def _create_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class FusionTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.cfg = config
        self.tcfg = config["train"]["phase2"]
        self.device = config["train"]["device"]
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            self.device = "cpu"

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        person_balanced = bool(config.get("data", {}).get("phase2", {}).get("person_balanced_sampling", False))
        logger.info(f"Phase2 trainer init: person_balanced_sampling={person_balanced}")

        self.class_weights = self._build_class_weights()
        self.cls_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.tcfg.get("label_smoothing", 0.0),
        )
        self.optimizer = self._get_optimizer(freeze=True)
        self.scheduler = _create_cosine_scheduler(
            self.optimizer,
            warmup_steps=len(train_loader) * self.tcfg.get("warmup_epochs", 2),
            total_steps=max(1, len(train_loader) * self.tcfg["num_epochs"]),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in self.device))

        out_dir = config["train"]["output_dir"]
        self.model_dir = os.path.join(out_dir, "models")
        self.report_dir = os.path.join(out_dir, "reports")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(out_dir, "tensorboard", "phase2"))
        self.best_macro_f1 = -1.0

    def _build_class_weights(self):
        if not self.tcfg.get("use_class_weight", True):
            return None
        labels = []
        for i in range(len(self.train_loader.dataset)):
            item = self.train_loader.dataset[i]
            labels.append(int(item["rpe"].item()))
        if not labels:
            return None
        num_classes = max(labels) + 1
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts[counts <= 0] = 1.0
        inv = 1.0 / counts
        weights = inv / inv.mean()
        cap = float(self.tcfg.get("class_weight_cap", 3.0))
        weights = np.clip(weights, 1.0 / cap, cap)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        logger.info(f"Phase2 class weights: {weights.tolist()} (counts={counts.astype(int).tolist()})")
        return w

    def _get_optimizer(self, freeze=True):
        for name, p in self.model.named_parameters():
            if "encoder" in name:
                p.requires_grad = not freeze

        lr = self.tcfg["lr"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if "encoder" not in n],
                "lr": lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "encoder" in n],
                "lr": lr * self.tcfg.get("encoder_lr_ratio", 0.1),
            },
        ]
        return AdamW(params, weight_decay=self.tcfg.get("weight_decay", 1e-4))

    def _compute_loss(self, batch, augment=True):
        emg = batch["emg"].to(self.device)
        imu = batch["imu"].to(self.device)
        labels = batch["rpe"].to(self.device)
        lw = self.tcfg["loss_weights"]

        out = self.model(emg, imu)
        loss_cls = self.cls_criterion(out["logits"], labels)

        loss_cont = torch.tensor(0.0, device=self.device)
        if augment:
            emg_v1, emg_v2 = augment_emg(emg), augment_emg(emg)
            imu_v1, imu_v2 = augment_imu(imu), augment_imu(imu)
            v1 = self.model(emg_v1, imu_v1)
            v2 = self.model(emg_v2, imu_v2)
            loss_cont = info_nce_symmetric(v1["fused_proj"], v2["fused_proj"]) * lw["cross_modal"]
            loss_cont += info_nce_symmetric(v1["emg_proj"], v2["emg_proj"]) * lw["emg_intra"]
            loss_cont += info_nce_symmetric(v1["imu_proj"], v2["imu_proj"]) * lw["imu_intra"]

        gate_balance_weight = float(self.tcfg.get("gate_balance_weight", 0.0))
        loss_gate = torch.tensor(0.0, device=self.device)
        if gate_balance_weight > 0.0:
            # 约束 batch 平均 gate 不要塌缩到单模态
            gate_mean = out["modality_gate"].mean(dim=0)
            target = torch.full_like(gate_mean, 0.5)
            loss_gate = F.mse_loss(gate_mean, target) * gate_balance_weight

        return loss_cls * lw["classification"] + loss_cont + loss_gate

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(self.train_loader, desc=f"Phase2 E{epoch+1} [Train]")
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=("cuda" in self.device)):
                loss = self._compute_loss(batch, augment=True)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, object]:
        self.model.eval()
        y_true, y_pred = [], []
        gate_values = []
        for batch in loader:
            labels = batch["rpe"].to(self.device)
            out = self.model(batch["emg"].to(self.device), batch["imu"].to(self.device))
            preds = out["logits"].argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            gate_values.append(out["modality_gate"].mean(dim=0).cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        cm = confusion_matrix(y_true, y_pred).tolist()
        gate_mean = (
            np.mean(np.stack(gate_values, axis=0), axis=0).tolist()
            if gate_values
            else [0.0, 0.0]
        )
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm,
            "modality_gate_mean": gate_mean,
        }

    def train(self):
        freeze_epochs = self.tcfg.get("freeze_epochs", 5)
        patience = self.tcfg.get("patience", 10)
        patience_counter = 0
        best_path = os.path.join(self.model_dir, "best_fusion_model.pt")

        for epoch in range(self.tcfg["num_epochs"]):
            if epoch == freeze_epochs:
                logger.info("解冻编码器，进入联合微调")
                self.optimizer = self._get_optimizer(freeze=False)

            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
            self.writer.add_scalar("val/gate_emg", val_metrics["modality_gate_mean"][0], epoch)
            self.writer.add_scalar("val/gate_imu", val_metrics["modality_gate_mean"][1], epoch)

            logger.info(
                f"[Phase2] E{epoch+1}/{self.tcfg['num_epochs']} "
                f"loss={train_loss:.4f} acc={val_metrics['accuracy']:.4f} "
                f"macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > self.best_macro_f1:
                self.best_macro_f1 = val_metrics["macro_f1"]
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.writer.close()
        logger.info(f"Best fusion model saved: {best_path}")

    def dump_report(self, test_metrics: Dict[str, object], filename: str = "phase2_eval.json"):
        path = os.path.join(self.report_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Phase2 report saved: {path}")
