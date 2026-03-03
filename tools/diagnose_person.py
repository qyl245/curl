"""
按受试者做 Phase2 失败诊断（默认 T456）。

输出:
- outputs/diagnostics/<person>/summary.json
- outputs/diagnostics/<person>/rep_level_stats.csv
- outputs/diagnostics/<person>/person_compare.csv
- outputs/diagnostics/<person>/fig_*.png
- outputs/diagnostics/<person>/conclusion.md
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


EMG_COL = "emg"
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
# 与 scripts/preprocess_jtom.py 保持一致的分组口径
GROUP_KEYS = ["person_id", "class", "rep_num"]


@dataclass
class Thresholds:
    coverage_ratio_red: float = 0.60
    zscore_red: float = 2.0
    zscore_metric_count_red: int = 3
    quality_sigma_red: float = 2.0
    boundary_focus_yellow: float = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose one target person (default: T456).")
    parser.add_argument("--csv_path", type=str, default="dataset/j-tom03.csv")
    parser.add_argument("--pt_path", type=str, default="dataset/processed/jtom_integrated_samples.pt")
    parser.add_argument("--target_person", type=str, default="T456")
    parser.add_argument("--fs_csv", type=float, default=2148.1481)
    parser.add_argument("--min_rows", type=int, default=50)
    parser.add_argument("--output_root", type=str, default="outputs/diagnostics")
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _nan_ratio(arr: np.ndarray) -> float:
    return float(np.isnan(arr).mean()) if arr.size else 0.0


def _constant_ratio(arr: np.ndarray, eps: float = 1e-8) -> float:
    if arr.size < 2:
        return 0.0
    diff = np.abs(np.diff(arr))
    return float((diff < eps).mean())


def _spike_ratio(arr: np.ndarray, z_thresh: float = 6.0) -> float:
    if arr.size < 5:
        return 0.0
    x = np.asarray(arr, dtype=np.float64)
    mu = np.nanmean(x)
    sigma = np.nanstd(x) + 1e-8
    z = np.abs((x - mu) / sigma)
    return float((z > z_thresh).mean())


def _cheating_index(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> float:
    std_x = float(np.nanstd(acc_x))
    std_y = float(np.nanstd(acc_y))
    std_z = float(np.nanstd(acc_z))
    return (std_x + std_z) / (std_y + 1e-6)


def _peak_location(signal_1d: np.ndarray) -> float:
    if signal_1d.size == 0:
        return 0.0
    idx = int(np.nanargmax(np.abs(signal_1d)))
    return float(idx / max(signal_1d.size - 1, 1))


def build_rep_features(df: pd.DataFrame, fs_csv: float, min_rows: int = 50) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    grouped = df.groupby(GROUP_KEYS, sort=False)
    for (pid, cls, rep), g in grouped:
        if len(g) < min_rows:
            continue

        emg = g[EMG_COL].to_numpy(dtype=np.float64, copy=False)
        acc_x = g["acc_x"].to_numpy(dtype=np.float64, copy=False)
        acc_y = g["acc_y"].to_numpy(dtype=np.float64, copy=False)
        acc_z = g["acc_z"].to_numpy(dtype=np.float64, copy=False)
        gyro_x = g["gyro_x"].to_numpy(dtype=np.float64, copy=False)
        gyro_y = g["gyro_y"].to_numpy(dtype=np.float64, copy=False)
        gyro_z = g["gyro_z"].to_numpy(dtype=np.float64, copy=False)

        n_rows = int(len(g))
        duration_s = float(n_rows / fs_csv)
        rpe = float(g["rpe"].iloc[0]) if n_rows else np.nan

        acc_norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        gyro_norm = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        jerk = float(np.nanstd(np.diff(acc_y))) * fs_csv if n_rows > 2 else 0.0

        row = {
            "person_id": str(pid),
            "class": int(cls),
            "rep_num": int(rep),
            # 这里仅做参考字段，不参与样本定义
            "weight_first": float(g["weight"].iloc[0]),
            "weight_nunique": int(g["weight"].nunique()),
            "rpe": rpe,
            "n_rows": n_rows,
            "duration_s": duration_s,
            "emg_mean_abs": float(np.nanmean(np.abs(emg))),
            "emg_std": float(np.nanstd(emg)),
            "emg_p2p": float(np.nanmax(emg) - np.nanmin(emg)),
            "acc_norm_mean": float(np.nanmean(acc_norm)),
            "acc_norm_std": float(np.nanstd(acc_norm)),
            "gyro_norm_mean": float(np.nanmean(gyro_norm)),
            "gyro_norm_std": float(np.nanstd(gyro_norm)),
            "jerk_score": jerk,
            "cheating_index": _cheating_index(acc_x, acc_y, acc_z),
            "peak_location": _peak_location(emg),
            "nan_ratio": np.mean(
                [
                    _nan_ratio(emg),
                    _nan_ratio(acc_x),
                    _nan_ratio(acc_y),
                    _nan_ratio(acc_z),
                    _nan_ratio(gyro_x),
                    _nan_ratio(gyro_y),
                    _nan_ratio(gyro_z),
                ]
            ),
            "constant_ratio": np.mean(
                [
                    _constant_ratio(emg),
                    _constant_ratio(acc_x),
                    _constant_ratio(acc_y),
                    _constant_ratio(acc_z),
                    _constant_ratio(gyro_x),
                    _constant_ratio(gyro_y),
                    _constant_ratio(gyro_z),
                ]
            ),
            "spike_ratio": np.mean(
                [
                    _spike_ratio(emg),
                    _spike_ratio(acc_x),
                    _spike_ratio(acc_y),
                    _spike_ratio(acc_z),
                    _spike_ratio(gyro_x),
                    _spike_ratio(gyro_y),
                    _spike_ratio(gyro_z),
                ]
            ),
        }
        rows.append(row)

    rep_df = pd.DataFrame(rows)
    if rep_df.empty:
        raise ValueError("No rep-level samples were built from CSV.")
    return rep_df


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / max(x.size + y.size - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def build_person_compare(rep_df: pd.DataFrame, target_person: str, metrics: List[str]) -> pd.DataFrame:
    target = rep_df[rep_df["person_id"] == target_person]
    others = rep_df[rep_df["person_id"] != target_person]
    if target.empty:
        raise ValueError(f"Target person {target_person} not found in rep table.")
    if others.empty:
        raise ValueError("No non-target samples found; cannot compare distributions.")

    rows = []
    for m in metrics:
        x = target[m].to_numpy(dtype=np.float64)
        y = others[m].to_numpy(dtype=np.float64)
        mean_t = float(np.mean(x))
        mean_o = float(np.mean(y))
        std_o = float(np.std(y) + 1e-8)
        z = float((mean_t - mean_o) / std_o)
        d = _cohen_d(x, y)
        rows.append(
            {
                "metric": m,
                "target_mean": mean_t,
                "others_mean": mean_o,
                "delta": mean_t - mean_o,
                "zscore_vs_others": z,
                "cohen_d": d,
            }
        )
    out = pd.DataFrame(rows).sort_values(by="zscore_vs_others", key=np.abs, ascending=False)
    return out


def _make_rpe_bin_2class(rpe: pd.Series) -> pd.Series:
    # 2-6: low, 7-10: high
    return np.where(rpe <= 6, "low_2_6", "high_7_10")


def _plot_rpe_distribution(rep_df: pd.DataFrame, target_person: str, out_path: str) -> None:
    target = rep_df[rep_df["person_id"] == target_person]
    others = rep_df[rep_df["person_id"] != target_person]
    x_labels = list(range(int(rep_df["rpe"].min()), int(rep_df["rpe"].max()) + 1))

    t_counts = target["rpe"].value_counts().reindex(x_labels, fill_value=0)
    o_counts = others["rpe"].value_counts().reindex(x_labels, fill_value=0)

    idx = np.arange(len(x_labels))
    width = 0.42
    plt.figure(figsize=(10, 5))
    plt.bar(idx - width / 2, t_counts.values, width=width, label=target_person)
    plt.bar(idx + width / 2, o_counts.values, width=width, label="others")
    plt.xticks(idx, x_labels)
    plt.xlabel("RPE")
    plt.ylabel("Count")
    plt.title("RPE Distribution: Target vs Others")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_duration_box(rep_df: pd.DataFrame, out_path: str) -> None:
    persons = sorted(rep_df["person_id"].unique().tolist())
    data = [rep_df.loc[rep_df["person_id"] == p, "duration_s"].values for p in persons]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=persons, showfliers=True)
    plt.xlabel("Person")
    plt.ylabel("Duration (s)")
    plt.title("Rep Duration by Person")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_feature_zscores(compare_df: pd.DataFrame, out_path: str) -> None:
    top = compare_df.head(12).copy()
    top = top.iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["metric"], top["zscore_vs_others"])
    plt.axvline(2.0, color="r", linestyle="--", linewidth=1)
    plt.axvline(-2.0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Z-score vs Others")
    plt.title("Top Feature Deviations (Target vs Others)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_person_class_heatmap(rep_df: pd.DataFrame, out_path: str) -> None:
    pivot = (
        rep_df.groupby(["person_id", "class"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    arr = pivot.values
    plt.figure(figsize=(10, 5))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="Rep Count")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.xlabel("Class")
    plt.ylabel("Person")
    plt.title("Person x Class Rep Coverage")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def check_pt_consistency(pt_path: str, expected_n_samples: Optional[int] = None) -> Dict[str, object]:
    if torch is None:
        return {"available": False, "reason": "torch not importable"}
    if not os.path.exists(pt_path):
        return {"available": False, "reason": f"missing: {pt_path}"}

    samples = torch.load(pt_path, map_location="cpu")
    people = [str(s["label"]["person_id"]) for s in samples]
    rpe = [int(s["label"]["rpe"]) for s in samples]
    out = {
        "available": True,
        "n_samples": int(len(samples)),
        "n_people": int(len(set(people))),
        "people": sorted(set(people)),
        "rpe_counts": {str(k): int(v) for k, v in pd.Series(rpe).value_counts().sort_index().items()},
    }
    if expected_n_samples is not None:
        out["expected_samples_from_csv"] = int(expected_n_samples)
        out["sample_count_match"] = bool(int(expected_n_samples) == int(len(samples)))
    return out


def _risk_level(flag: bool) -> str:
    return "red" if flag else "green"


def build_summary(
    rep_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    target_person: str,
    pt_info: Dict[str, object],
    thresholds: Thresholds,
) -> Dict[str, object]:
    person_counts = rep_df["person_id"].value_counts().to_dict()
    target_n = int(person_counts.get(target_person, 0))
    others_counts = [v for k, v in person_counts.items() if k != target_person]
    others_median = float(np.median(others_counts)) if others_counts else 0.0
    coverage_ratio = float(target_n / max(others_median, 1.0))
    coverage_red = coverage_ratio < thresholds.coverage_ratio_red

    quality_cols = ["nan_ratio", "constant_ratio", "spike_ratio"]
    quality_person = rep_df.groupby("person_id")[quality_cols].mean()
    target_quality = quality_person.loc[target_person]
    quality_mean = quality_person.mean()
    quality_std = quality_person.std().replace(0, 1e-8)
    quality_z = (target_quality - quality_mean) / quality_std
    quality_red = bool((quality_z > thresholds.quality_sigma_red).any())

    abs_z = compare_df["zscore_vs_others"].abs()
    z_red_count = int((abs_z > thresholds.zscore_red).sum())
    shift_red = z_red_count >= thresholds.zscore_metric_count_red

    rep_df = rep_df.copy()
    rep_df["rpe_bin"] = _make_rpe_bin_2class(rep_df["rpe"])
    tgt = rep_df[rep_df["person_id"] == target_person]
    bin_counts = tgt["rpe_bin"].value_counts().to_dict()
    total_tgt = max(int(len(tgt)), 1)
    # 近似“边界敏感”: 看是否大量样本集中在 6/7 附近
    boundary_focus = float(((tgt["rpe"] == 6) | (tgt["rpe"] == 7)).mean())
    boundary_yellow = boundary_focus > thresholds.boundary_focus_yellow

    risk_items = [
        {
            "name": "coverage_insufficient",
            "value": coverage_ratio,
            "threshold": thresholds.coverage_ratio_red,
            "status": _risk_level(coverage_red),
            "detail": f"target_n={target_n}, others_median={others_median:.1f}",
        },
        {
            "name": "distribution_shift",
            "value": z_red_count,
            "threshold": thresholds.zscore_metric_count_red,
            "status": _risk_level(shift_red),
            "detail": f"|z|>{thresholds.zscore_red} metrics={z_red_count}",
        },
        {
            "name": "signal_quality_risk",
            "value": float(quality_z.max()),
            "threshold": thresholds.quality_sigma_red,
            "status": _risk_level(quality_red),
            "detail": f"quality_z={quality_z.to_dict()}",
        },
        {
            "name": "boundary_sensitive",
            "value": boundary_focus,
            "threshold": thresholds.boundary_focus_yellow,
            "status": "yellow" if boundary_yellow else "green",
            "detail": "ratio of samples at RPE 6/7",
        },
    ]

    if any(x["status"] == "red" for x in risk_items):
        overall = "red"
    elif any(x["status"] == "yellow" for x in risk_items):
        overall = "yellow"
    else:
        overall = "green"

    top_shift = compare_df.head(5)[["metric", "zscore_vs_others", "cohen_d"]].to_dict(orient="records")

    summary = {
        "target_person": target_person,
        "overall_status": overall,
        "n_reps_total": int(len(rep_df)),
        "n_persons": int(rep_df["person_id"].nunique()),
        "person_counts": {str(k): int(v) for k, v in person_counts.items()},
        "target_rpe_counts": {str(k): int(v) for k, v in tgt["rpe"].value_counts().sort_index().items()},
        "target_rpe_2class_counts": {str(k): int(v) for k, v in bin_counts.items()},
        "risk_items": risk_items,
        "top_shift_metrics": top_shift,
        "pt_consistency": pt_info,
    }
    return summary


def write_conclusion_md(summary: Dict[str, object], compare_df: pd.DataFrame, out_path: str) -> None:
    risk_items = summary["risk_items"]  # type: ignore[index]
    reds = [x for x in risk_items if x["status"] == "red"]  # type: ignore[index]
    yellows = [x for x in risk_items if x["status"] == "yellow"]  # type: ignore[index]

    lines = []
    lines.append(f"# Diagnostic Conclusion: {summary['target_person']}")
    lines.append("")
    lines.append(f"- Overall status: **{summary['overall_status']}**")
    lines.append(f"- Persons: {summary['n_persons']}, reps: {summary['n_reps_total']}")
    lines.append("")
    lines.append("## Risk checklist")
    for item in risk_items:
        lines.append(
            f"- {item['name']}: {item['status']} "
            f"(value={item['value']}, threshold={item['threshold']}) | {item['detail']}"
        )
    lines.append("")
    lines.append("## Top shifted features")
    top = compare_df.head(8)
    for _, r in top.iterrows():
        lines.append(
            f"- {r['metric']}: z={r['zscore_vs_others']:.3f}, d={r['cohen_d']:.3f}, "
            f"delta={r['delta']:.6f}"
        )
    lines.append("")
    lines.append("## Suggested next action")
    if reds:
        lines.append("- Prioritize data-level fix first (coverage / quality / distribution alignment).")
        lines.append("- Re-run LOO after cleaning before touching model architecture.")
    elif yellows:
        lines.append("- Likely boundary sensitivity or mild shift; test label boundary and sampling strategy.")
        lines.append("- Then run lightweight training ablations.")
    else:
        lines.append("- Data risks are not obvious; likely model capacity/training objective bottleneck.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)

    out_dir = os.path.join(args.output_root, args.target_person)
    _safe_mkdir(out_dir)

    use_cols = GROUP_KEYS + ["rpe", EMG_COL] + ACC_COLS + GYRO_COLS
    # 额外读取 weight，用于辅助观测（不参与分组）
    use_cols = ["weight"] + use_cols
    df = pd.read_csv(args.csv_path, usecols=use_cols)
    rep_df = build_rep_features(df, fs_csv=args.fs_csv, min_rows=args.min_rows)

    metrics = [
        "duration_s",
        "emg_mean_abs",
        "emg_std",
        "emg_p2p",
        "acc_norm_mean",
        "acc_norm_std",
        "gyro_norm_mean",
        "gyro_norm_std",
        "jerk_score",
        "cheating_index",
        "peak_location",
        "nan_ratio",
        "constant_ratio",
        "spike_ratio",
    ]
    compare_df = build_person_compare(rep_df, args.target_person, metrics)
    pt_info = check_pt_consistency(args.pt_path, expected_n_samples=len(rep_df))
    summary = build_summary(
        rep_df=rep_df,
        compare_df=compare_df,
        target_person=args.target_person,
        pt_info=pt_info,
        thresholds=Thresholds(),
    )

    rep_csv_path = os.path.join(out_dir, "rep_level_stats.csv")
    cmp_csv_path = os.path.join(out_dir, "person_compare.csv")
    summary_json_path = os.path.join(out_dir, "summary.json")
    conclusion_md_path = os.path.join(out_dir, "conclusion.md")

    rep_df.to_csv(rep_csv_path, index=False, encoding="utf-8")
    compare_df.to_csv(cmp_csv_path, index=False, encoding="utf-8")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _plot_rpe_distribution(rep_df, args.target_person, os.path.join(out_dir, "fig_rpe_distribution.png"))
    _plot_duration_box(rep_df, os.path.join(out_dir, "fig_duration_boxplot.png"))
    _plot_feature_zscores(compare_df, os.path.join(out_dir, "fig_feature_zscores.png"))
    _plot_person_class_heatmap(rep_df, os.path.join(out_dir, "fig_person_class_heatmap.png"))
    write_conclusion_md(summary, compare_df, conclusion_md_path)

    print("Diagnosis finished.")
    print(f"Output directory: {out_dir}")
    print(f"Overall status: {summary['overall_status']}")


if __name__ == "__main__":
    main()
