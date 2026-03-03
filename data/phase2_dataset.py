"""
Phase 2 融合数据集：jtom_integrated_samples.pt，按 person 划分 train/val/test
"""
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.logging_utils import setup_logger

logger = setup_logger()


class Phase2RepDataset(Dataset):
    def __init__(self, samples: List[Dict], rpe_bins: List[float]):
        self.samples = samples
        self.rpe_bins = rpe_bins

    def __len__(self) -> int:
        return len(self.samples)

    def _bucketize(self, rpe: float) -> int:
        for i in range(len(self.rpe_bins) - 1):
            left, right = self.rpe_bins[i], self.rpe_bins[i + 1]
            if left <= rpe < right:
                return i
        return len(self.rpe_bins) - 2

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        label = s["label"]
        rpe_raw = float(label["rpe"])
        return {
            "emg": s["emg"].float(),
            "imu": s["imu"].float(),
            "rpe": torch.tensor(self._bucketize(rpe_raw), dtype=torch.long),
            "rpe_raw": torch.tensor(rpe_raw, dtype=torch.float32),
        }


def _build_person_balanced_sampler(samples: List[Dict], seed: int):
    """按 person_id 反频次加权采样，缓解小样本人群被淹没。"""
    person_ids = [str(s["label"]["person_id"]) for s in samples]
    counts = Counter(person_ids)
    weights = [1.0 / counts[pid] for pid in person_ids]
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
        generator=generator,
    )
    return sampler, counts


def _split_people(samples: List[Dict], config: dict) -> Tuple[set, set, set]:
    split_cfg = config["data"]["phase2"]["split"]
    train_people = split_cfg.get("train_persons", [])
    val_people = split_cfg.get("val_persons", [])
    test_people = split_cfg.get("test_persons", [])

    if train_people and val_people and test_people:
        return set(map(str, train_people)), set(map(str, val_people)), set(map(str, test_people))

    ratios = split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    people = sorted({str(s["label"]["person_id"]) for s in samples})
    rng = np.random.RandomState(config["train"].get("seed", 42))
    rng.shuffle(people)
    n = len(people)
    n_train = max(1, int(n * ratios["train"]))
    n_val = max(1, int(n * ratios["val"]))
    train_set = set(people[:n_train])
    val_set = set(people[n_train : n_train + n_val])
    test_set = set(people[n_train + n_val :]) or set(people[-1:])
    return train_set, val_set, test_set


def create_phase2_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dcfg = config["data"]["phase2"]
    pt_path = dcfg["processed_pt_path"]
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"Phase2 样本文件不存在: {pt_path}. 请先运行 scripts/preprocess_jtom.py"
        )

    samples: List[Dict] = torch.load(pt_path)
    train_people, val_people, test_people = _split_people(samples, config)

    train_samples = [s for s in samples if str(s["label"]["person_id"]) in train_people]
    val_samples = [s for s in samples if str(s["label"]["person_id"]) in val_people]
    test_samples = [s for s in samples if str(s["label"]["person_id"]) in test_people]

    rpe_bins = dcfg["rpe_bins"]
    logger.info(
        "Phase2 split by person: "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )

    bs = dcfg["batch_size"]
    num_workers = config["train"].get("num_workers", 0)
    pin_memory = bool(config["train"].get("pin_memory", True))
    seed = int(config["train"].get("seed", 42))
    person_balanced = bool(dcfg.get("person_balanced_sampling", False))

    train_dataset = Phase2RepDataset(train_samples, rpe_bins)
    if person_balanced and len(train_samples) > 0:
        sampler, person_counts = _build_person_balanced_sampler(train_samples, seed=seed)
        logger.info(
            "Phase2 person-balanced sampling enabled: "
            f"counts={dict(sorted(person_counts.items()))}, samples_per_epoch={len(train_samples)}"
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        Phase2RepDataset(val_samples, rpe_bins),
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Phase2RepDataset(test_samples, rpe_bins),
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def get_num_phase2_classes(config: dict) -> int:
    bins = config["data"]["phase2"]["rpe_bins"]
    return len(bins) - 1
