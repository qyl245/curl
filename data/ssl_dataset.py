"""
Phase 1 SSL 数据集：MyoGym (EMG) + imu_fatigue (IMU)
"""
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from dataloaders.augmentations import create_ssl_transform
from utils.logging_utils import setup_logger

logger = setup_logger()

EMG_RAW_COLS = [f"emg_{i}" for i in range(1, 9)]
IMU_RAW_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def butter_lowpass_filter(data: np.ndarray, cutoff=5.0, fs=50.0, order=4) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def preprocess_emg(df: pd.DataFrame) -> np.ndarray:
    from sklearn.decomposition import PCA
    raw_emg = df[EMG_RAW_COLS].values.astype(np.float32)
    emg_detrended = raw_emg - np.mean(raw_emg, axis=0, keepdims=True)
    emg_rectified = np.abs(emg_detrended)
    emg_envelope = butter_lowpass_filter(emg_rectified, cutoff=5.0, fs=50.0)
    pca = PCA(n_components=1)
    return pca.fit_transform(emg_envelope).astype(np.float32)


def preprocess_imu(df: pd.DataFrame) -> np.ndarray:
    from scipy.signal import medfilt
    imu_df = df[IMU_RAW_COLS].copy()
    for col in IMU_RAW_COLS:
        imu_df[col] = pd.to_numeric(imu_df[col], errors="coerce")

    if imu_df.isna().values.any():
        nan_before = int(imu_df.isna().sum().sum())
        logger.warning(f"IMU 原始数据存在缺失值: {nan_before}，将执行插值与填补。")

    if "person_id" in df.columns:
        imu_df = (
            imu_df.assign(person_id=df["person_id"].astype(str).values)
            .groupby("person_id", group_keys=False)[IMU_RAW_COLS]
            .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
        )
    imu_df = imu_df.ffill().bfill().fillna(0.0)

    raw_imu = imu_df[IMU_RAW_COLS].values.astype(np.float32)
    imu_denoised = medfilt(raw_imu, kernel_size=(3, 1)).astype(np.float32)
    return np.nan_to_num(imu_denoised, nan=0.0, posinf=0.0, neginf=0.0)


class SSLDataset(Dataset):
    def __init__(self, windows: list, transform=None):
        self.windows = windows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        import torch
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


def _window_by_person(
    data_mat: np.ndarray,
    person_ids: np.ndarray,
    window_size: int,
    step_size: int,
) -> list:
    all_windows = []
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


def create_ssl_dataloaders(config: dict, modality: str) -> tuple:
    from sklearn.preprocessing import StandardScaler

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
