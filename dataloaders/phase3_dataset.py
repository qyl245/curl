"""
Phase 3 数据集：coaching.jsonl + jtom_integrated_samples.pt + analyse_jtom.csv
signal_id 格式: {person_id}_W{weight}_C{class}_R{rep_num}_v{0|1|2}
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.logging_utils import setup_logger

logger = setup_logger()


def _parse_signal_id(signal_id: str) -> Tuple[str, int, int, int]:
    """
    解析 signal_id 为 (person_id, weight, class, rep_num)。
    例: A321_W5_C1_R1_v0 -> ("A321", 5, 1, 1)
    """
    m = re.match(r"^(.+)_W(\d+)_C(\d+)_R(\d+)_v\d+$", signal_id)
    if not m:
        raise ValueError(f"无法解析 signal_id: {signal_id}")
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _format_stat_text(row: pd.Series) -> str:
    """将 analyse_jtom 单行格式化为 Layer 2 统计指标文本。"""
    return (
        f"借力指数: {row['cheating_index']:.2f}, "
        f"震颤: {row['tremor_score']:.2f}, "
        f"时长: {row['duration']:.2f}s, "
        f"流畅度: {row['jerk_score']:.2f}, "
        f"EMG强度: {row['emg_intensity']:.4f}, "
        f"RPE: {row['rpe']:.0f}"
    )


class Phase3Dataset(Dataset):
    """
    Phase 3 训练数据集。
    每个样本: emg, imu, stat_text, target_text
    """

    def __init__(
        self,
        coaching_path: str,
        jtom_pt_path: str,
        analyse_jtom_path: str,
        val_ratio: float = 0.1,
        seed: int = 42,
        split: str = "train",
    ):
        """
        Args:
            coaching_path: coaching.jsonl 路径
            jtom_pt_path: jtom_integrated_samples.pt 路径
            analyse_jtom_path: analyse_jtom.csv 路径
            val_ratio: 验证集比例
            seed: 随机种子
            split: "train" 或 "val"
        """
        self.split = split

        # 1. 加载 coaching.jsonl
        coaching_rows = []
        with open(coaching_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    coaching_rows.append(json.loads(line))

        # 2. 加载 jtom 样本并建索引
        # full_key: (person_id, weight, class, rep_num) -> idx（推荐口径）
        # loose_key: (person_id, class, rep_num) -> [idx...]（兼容历史 PT 未按 weight 分组）
        samples = torch.load(jtom_pt_path, map_location="cpu", weights_only=False)
        self.samples = samples
        jtom_index: Dict[Tuple[str, int, int, int], int] = {}
        jtom_loose_index: Dict[Tuple[str, int, int], List[int]] = {}
        for idx, s in enumerate(samples):
            lbl = s["label"]
            key = (
                str(lbl["person_id"]),
                int(lbl["weight"]),
                int(lbl["class"]),
                int(lbl["rep_num"]),
            )
            jtom_index[key] = idx
            loose_key = (str(lbl["person_id"]), int(lbl["class"]), int(lbl["rep_num"]))
            jtom_loose_index.setdefault(loose_key, []).append(idx)

        # 3. 加载 analyse_jtom 并建索引
        df = pd.read_csv(analyse_jtom_path)
        df["key"] = list(
            zip(
                df["person_id"].astype(str),
                df["weight"].astype(int),
                df["class"].astype(int),
                df["rep_num"].astype(int),
            )
        )
        stats_index = df.set_index("key").to_dict("index")

        # 4. 构建有效样本
        self.rows: List[Dict] = []
        missing_jtom = 0
        missing_stats = 0
        fallback_jtom = 0
        ambiguous_jtom = 0
        for r in coaching_rows:
            try:
                pid, w, c, rep = _parse_signal_id(r["signal_id"])
                key = (pid, w, c, rep)
                if key in jtom_index:
                    jtom_idx = jtom_index[key]
                else:
                    # 兼容历史 PT（未按 weight 分组）: 尝试 (person_id, class, rep_num) 回退匹配
                    loose_key = (pid, c, rep)
                    cands = jtom_loose_index.get(loose_key, [])
                    if len(cands) == 1:
                        jtom_idx = cands[0]
                        fallback_jtom += 1
                    elif len(cands) > 1:
                        ambiguous_jtom += 1
                        continue
                    else:
                        missing_jtom += 1
                        continue
                if key not in stats_index:
                    missing_stats += 1
                    continue
                self.rows.append({
                    "signal_id": r["signal_id"],
                    "target_text": r["text"],
                    "jtom_idx": jtom_idx,
                    "stat_row": stats_index[key],
                })
            except (ValueError, KeyError) as e:
                continue

        if missing_jtom > 0:
            logger.warning(f"Phase3: {missing_jtom} 条 coaching 无对应 jtom 样本")
        if fallback_jtom > 0:
            logger.warning(f"Phase3: {fallback_jtom} 条 coaching 使用了 jtom 回退匹配(忽略 weight)")
        if ambiguous_jtom > 0:
            logger.warning(f"Phase3: {ambiguous_jtom} 条 coaching 命中多个 jtom 候选(按 loose_key)，已跳过")
        if missing_stats > 0:
            logger.warning(f"Phase3: {missing_stats} 条 coaching 无对应 analyse_jtom 行")

        # 5. 按 rep 划分 train/val，保证同一 rep 的 v0/v1/v2 同属一 split
        import numpy as np

        def _rep_key(r):
            s = self.samples[r["jtom_idx"]]
            lbl = s["label"]
            return (str(lbl["person_id"]), int(lbl["weight"]), int(lbl["class"]), int(lbl["rep_num"]))

        rep_keys = list({_rep_key(r) for r in self.rows})
        rng = np.random.RandomState(seed)
        rng.shuffle(rep_keys)
        n_val = max(1, int(len(rep_keys) * val_ratio))
        val_keys = set(rep_keys[:n_val])
        train_keys = set(rep_keys[n_val:])

        target_keys = val_keys if split == "val" else train_keys
        self.rows = [r for r in self.rows if _rep_key(r) in target_keys]

        logger.info(f"Phase3 {split}: {len(self.rows)} 条样本")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        s = self.samples[r["jtom_idx"]]
        stat_text = _format_stat_text(pd.Series(r["stat_row"]))
        return {
            "emg": s["emg"].float(),
            "imu": s["imu"].float(),
            "stat_text": stat_text,
            "target_text": r["target_text"],
        }


def create_phase3_datasets(config: dict) -> Tuple[Phase3Dataset, Phase3Dataset]:
    """创建 Phase 3 train/val 数据集。"""
    p3 = config.get("phase3", {})
    data_cfg = config.get("data", {}).get("phase3", {})
    coaching_path = data_cfg.get("coaching_path", "dataset/coaching.jsonl")
    jtom_pt_path = config["data"]["phase2"]["processed_pt_path"]
    analyse_path = data_cfg.get("analyse_jtom_path", "dataset/analyse_jtom.csv")
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    seed = int(config.get("train", {}).get("seed", 42))

    train_ds = Phase3Dataset(
        coaching_path=coaching_path,
        jtom_pt_path=jtom_pt_path,
        analyse_jtom_path=analyse_path,
        val_ratio=val_ratio,
        seed=seed,
        split="train",
    )
    val_ds = Phase3Dataset(
        coaching_path=coaching_path,
        jtom_pt_path=jtom_pt_path,
        analyse_jtom_path=analyse_path,
        val_ratio=val_ratio,
        seed=seed,
        split="val",
    )
    return train_ds, val_ds
