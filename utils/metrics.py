"""
评估指标
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    return accuracy_score(labels.cpu().numpy(), preds)


def compute_f1(logits: torch.Tensor, labels: torch.Tensor, average="macro") -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    return f1_score(labels.cpu().numpy(), preds, average=average, zero_division=0)


def full_report(logits: torch.Tensor, labels: torch.Tensor, class_names=None) -> str:
    preds = logits.argmax(dim=-1).cpu().numpy()
    return classification_report(
        labels.cpu().numpy(), preds, target_names=class_names, zero_division=0
    )


def compute_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    preds = logits.argmax(dim=-1).cpu().numpy()
    return confusion_matrix(labels.cpu().numpy(), preds)
