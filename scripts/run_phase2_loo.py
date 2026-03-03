"""
Phase 2 留一人交叉验证（Leave-One-Person-Out）。
5 位受试者轮换作为 test 集，得到更稳定的 Macro-F1 评估。
"""
import copy
import json
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from utils.config import load_config
from utils.logging_utils import setup_logger

logger = setup_logger()


def get_people_from_pt(pt_path: str) -> list:
    """从 PT 文件中提取所有 person_id。"""
    samples = torch.load(pt_path, map_location="cpu")
    people = sorted({str(s["label"]["person_id"]) for s in samples})
    return people


def run_one_fold(config: dict, train_people: list, val_people: list, test_people: list, fold_idx: int):
    """运行单折训练与评估。"""
    from dataloaders.create_dataset import create_phase2_dataloaders, get_num_phase2_classes
    from models.fusion import MultiModalModel
    from training.fusion_trainer import FusionTrainer

    from main import build_encoder, _load_ckpt_if_exists

    cfg = copy.deepcopy(config)
    cfg["data"]["phase2"]["split"]["train_persons"] = train_people
    cfg["data"]["phase2"]["split"]["val_persons"] = val_people
    cfg["data"]["phase2"]["split"]["test_persons"] = test_people

    # 每折使用独立 output 子目录，避免互相覆盖
    base_out = cfg["train"]["output_dir"]
    cfg["train"]["output_dir"] = os.path.join(base_out, "loo", f"fold_{fold_idx}")

    train_loader, val_loader, test_loader = create_phase2_dataloaders(cfg)
    num_classes = get_num_phase2_classes(cfg)

    emg_encoder = build_encoder(cfg, "emg")
    imu_encoder = build_encoder(cfg, "imu")
    _load_ckpt_if_exists(emg_encoder, cfg["paths"]["phase1_emg_ckpt"], "EMG encoder")
    _load_ckpt_if_exists(imu_encoder, cfg["paths"]["phase1_imu_ckpt"], "IMU encoder")

    fusion_model = MultiModalModel(
        emg_encoder=emg_encoder,
        imu_encoder=imu_encoder,
        model_cfg=cfg["model"],
        num_classes=num_classes,
    )
    trainer = FusionTrainer(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
    )
    trainer.train()

    best_path = os.path.join(cfg["train"]["output_dir"], "models", "best_fusion_model.pt")
    if os.path.exists(best_path):
        fusion_model.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
        fusion_model.to(trainer.device)

    return trainer.evaluate(test_loader)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 留一人交叉验证")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["train"]["device"] = args.device
    config["train"]["seed"] = args.seed

    # 设置随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pt_path = config["data"]["phase2"]["processed_pt_path"]
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PT 文件不存在: {pt_path}，请先在云主机上运行预处理。")

    people = get_people_from_pt(pt_path)
    logger.info(f"共 {len(people)} 位受试者: {people}")

    if len(people) < 2:
        raise ValueError("至少需要 2 位受试者才能进行 LOO。")

    all_metrics = []
    rng = np.random.RandomState(args.seed)

    for fold_idx, test_person in enumerate(people):
        remaining = [p for p in people if p != test_person]
        rng.shuffle(remaining)
        val_person = remaining[-1]
        train_people = remaining[:-1]

        logger.info(f"===== Fold {fold_idx + 1}/{len(people)}: test={test_person}, val={val_person} =====")
        metrics = run_one_fold(
            config,
            train_people=train_people,
            val_people=[val_person],
            test_people=[test_person],
            fold_idx=fold_idx,
        )
        metrics["test_person"] = test_person
        all_metrics.append(metrics)
        logger.info(f"Fold {fold_idx + 1} macro_f1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f}")

    # 汇总
    macro_f1s = [m["macro_f1"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]
    num_classes = len(all_metrics[0]["per_class_recall"])
    per_class_recalls = [
        np.mean([m["per_class_recall"][c] for m in all_metrics])
        for c in range(num_classes)
    ]

    report = {
        "n_folds": len(people),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "per_class_recall_mean": per_class_recalls,
        "per_fold": [
            {"test_person": m["test_person"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
            for m in all_metrics
        ],
    }

    out_dir = config["train"]["output_dir"]
    report_path = os.path.join(out_dir, "reports", "phase2_loo_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"LOO 报告已保存: {report_path}")
    logger.info(f"Macro-F1: {report['macro_f1_mean']:.4f} ± {report['macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()
