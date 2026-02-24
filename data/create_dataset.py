import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, medfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.augmentations import create_ssl_transform
from utils.logging_utils import setup_logger

logger = setup_logger()

EMG_RAW_COLS = [f"emg_{i}" for i in range(1, 9)]
IMU_RAW_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def butter_lowpass_filter(data: np.ndarray, cutoff=5.0, fs=50.0, order=4) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def preprocess_emg(df: pd.DataFrame) -> np.ndarray:
    raw_emg = df[EMG_RAW_COLS].values.astype(np.float32)
    emg_detrended = raw_emg - np.mean(raw_emg, axis=0, keepdims=True)
    emg_rectified = np.abs(emg_detrended)
    emg_envelope = butter_lowpass_filter(emg_rectified, cutoff=5.0, fs=50.0)
    pca = PCA(n_components=1)
    return pca.fit_transform(emg_envelope).astype(np.float32)


def preprocess_imu(df: pd.DataFrame) -> np.ndarray:
    imu_df = df[IMU_RAW_COLS].copy()
    for col in IMU_RAW_COLS:
        imu_df[col] = pd.to_numeric(imu_df[col], errors="coerce")

    if imu_df.isna().values.any():
        nan_before = int(imu_df.isna().sum().sum())
        logger.warning(f"IMU 原始数据存在缺失值: {nan_before}，将执行插值与填补。")

    # 先按受试者做插值，再用全局前后向填补，最后兜底 0
    if "person_id" in df.columns:
        imu_df = (
            imu_df.assign(person_id=df["person_id"].astype(str).values)
            .groupby("person_id", group_keys=False)[IMU_RAW_COLS]
            .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
        )
    imu_df = imu_df.ffill().bfill().fillna(0.0)

    raw_imu = imu_df[IMU_RAW_COLS].values.astype(np.float32)
    imu_denoised = medfilt(raw_imu, kernel_size=(3, 1)).astype(np.float32)
    imu_denoised = np.nan_to_num(imu_denoised, nan=0.0, posinf=0.0, neginf=0.0)
    return imu_denoised


class SSLDataset(Dataset):
    def __init__(self, windows: List[Dict], transform=None):
        self.windows = windows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        win = self.windows[idx]
        sample = {
            "signal": torch.tensor(win["signal"], dtype=torch.float32),
            "person_id": win["person_id"],
        }
        if self.transform:
            sample = self.transform(sample)
        sample.pop("signal", None)
        sample.pop("person_id", None)
        return sample


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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        label = s["label"]
        rpe_raw = float(label["rpe"])
        return {
            "emg": s["emg"].float(),
            "imu": s["imu"].float(),
            "rpe": torch.tensor(self._bucketize(rpe_raw), dtype=torch.long),
            "rpe_raw": torch.tensor(rpe_raw, dtype=torch.float32),
        }


def _window_by_person(
    data_mat: np.ndarray,
    person_ids: np.ndarray,
    window_size: int,
    step_size: int,
) -> List[Dict]:
    all_windows: List[Dict] = []
    tmp = pd.DataFrame({"person_id": person_ids})
    for pid, group in tmp.groupby("person_id"):
        idx = group.index.to_numpy()
        arr = data_mat[idx]
        T = arr.shape[0]
        if T < window_size:
            continue
        for start in range(0, T - window_size + 1, step_size):
            end = start + window_size
            all_windows.append({"signal": arr[start:end].T, "person_id": pid})
    return all_windows


def create_ssl_dataloaders(config: dict, modality: str) -> Tuple[DataLoader, DataLoader]:
    dcfg = config["data"]["phase1"][modality]
    tcfg = config["train"]["phase1"]
    modality_upper = modality.upper()

    df = pd.read_csv(dcfg["csv_path"])
    if modality == "emg":
        processed = preprocess_emg(df)
    else:
        processed = preprocess_imu(df)

    scaler = StandardScaler()
    processed = scaler.fit_transform(processed).astype(np.float32)
    if not np.isfinite(processed).all():
        bad = int((~np.isfinite(processed)).sum())
        logger.warning(f"{modality_upper} 标准化后存在非有限值: {bad}，将强制置零。")
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
    person_ids = df["person_id"].astype(str).values

    windows = _window_by_person(
        processed,
        person_ids,
        window_size=dcfg["window_size"],
        step_size=dcfg["step_size"],
    )

    rng = np.random.RandomState(config["train"].get("seed", 42))
    all_pids = sorted({w["person_id"] for w in windows})
    rng.shuffle(all_pids)
    val_count = max(1, int(len(all_pids) * dcfg.get("val_ratio", 0.2)))
    val_pids = set(all_pids[:val_count])

    train_windows = [w for w in windows if w["person_id"] not in val_pids]
    val_windows = [w for w in windows if w["person_id"] in val_pids]

    logger.info(
        f"{modality_upper} SSL windows: total={len(windows)} "
        f"train={len(train_windows)} val={len(val_windows)}"
    )

    transform = create_ssl_transform(
        mask_ratio=tcfg["mask_ratio"],
        num_views=tcfg["num_views"],
        num_chunks=tcfg["num_chunks"],
        task_type=modality_upper,
    )
    num_workers = config["train"].get("num_workers", 0)
    pin_memory = bool(config["train"].get("pin_memory", True))

    train_loader = DataLoader(
        SSLDataset(train_windows, transform=transform),
        batch_size=dcfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        SSLDataset(val_windows, transform=transform),
        batch_size=dcfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


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
    val_set = set(people[n_train:n_train + n_val])
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

    train_loader = DataLoader(
        Phase2RepDataset(train_samples, rpe_bins),
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
