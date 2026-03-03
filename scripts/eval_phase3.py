"""
Phase3 评估脚本（最小可用版）

输入:
- dataset/coaching.jsonl          (GT: signal_id, tags, text)
- outputs/phase3_preds.jsonl      (Pred: signal_id, pred_text)
- dataset/analyse_jtom.csv        (可选，用于冲突率检测)

输出:
- outputs/reports/phase3_eval.json
- outputs/reports/phase3_eval_by_person.csv
- outputs/reports/phase3_badcases.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase3 text outputs.")
    parser.add_argument("--gt_path", type=str, default="dataset/coaching.jsonl")
    parser.add_argument("--pred_path", type=str, default="outputs/phase3_preds.jsonl")
    parser.add_argument("--analyse_path", type=str, default="dataset/analyse_jtom.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/reports")
    parser.add_argument("--eval_scope", type=str, default="matched", choices=["matched", "all"], help="matched: 仅对有预测样本计分；all: 全GT计分（缺失预测按漏检计）。")
    parser.add_argument("--no_collapse_versions", action="store_true", help="不聚合 v0/v1/v2；默认聚合为 rep 级评估。")
    return parser.parse_args()


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_signal_id(signal_id: str) -> Tuple[str, int, int, int]:
    m = re.match(r"^(.+)_W(\d+)_C(\d+)_R(\d+)_v\d+$", signal_id)
    if not m:
        raise ValueError(f"bad signal_id: {signal_id}")
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _base_signal_id(signal_id: str) -> str:
    return re.sub(r"_v\d+$", "", signal_id)


TAG_PATTERNS: Dict[str, List[str]] = {
    "借力": [r"借力", r"惯性", r"甩"],
    "晃动": [r"晃", r"躯干", r"摆动"],
    "后程困难": [r"后程", r"最后几个", r"顶不动", r"发力困难"],
    "震颤": [r"抖", r"震颤"],
    "标准": [r"标准", r"教科书", r"动作好", r"很稳", r"稳定"],
    "挣扎": [r"挣扎", r"吃力", r"费劲"],
}


def _extract_pred_tags(text: str) -> Set[str]:
    found: Set[str] = set()
    if not isinstance(text, str):
        return found
    for tag, pats in TAG_PATTERNS.items():
        for p in pats:
            if re.search(p, text):
                found.add(tag)
                break
    return found


def _normalize_gt_tags(raw_tags: str) -> Set[str]:
    if not isinstance(raw_tags, str):
        return set()
    src = [t.strip() for t in raw_tags.split("/") if t.strip()]
    out: Set[str] = set()
    for t in src:
        if "借力" in t:
            out.add("借力")
        elif "晃" in t or "躯干" in t or "摆动" in t:
            out.add("晃动")
        elif "后程" in t:
            out.add("后程困难")
        elif "震颤" in t or "抖" in t:
            out.add("震颤")
        elif "教科书" in t or "标准" in t:
            out.add("标准")
        elif "挣扎" in t or "费劲" in t or "吃力" in t:
            out.add("挣扎")
    return out


def _lcs_len(a: List[str], b: List[str]) -> int:
    # 轻量实现 ROUGE-L 的 LCS
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_f1(pred: str, ref: str) -> float:
    p = [c for c in str(pred).strip() if not c.isspace()]
    r = [c for c in str(ref).strip() if not c.isspace()]
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    prec = lcs / max(len(p), 1)
    rec = lcs / max(len(r), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def f1(self) -> float:
        p = self.tp / max(self.tp + self.fp, 1)
        r = self.tp / max(self.tp + self.fn, 1)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def _build_analyse_index(path: str):
    if not os.path.exists(path):
        return None, {}
    df = pd.read_csv(path)
    df["key"] = list(
        zip(
            df["person_id"].astype(str),
            df["weight"].astype(int),
            df["class"].astype(int),
            df["rep_num"].astype(int),
        )
    )
    index = df.set_index("key").to_dict("index")
    # 用分位数定义“高风险指标”
    q = {
        "tremor_score": float(df["tremor_score"].quantile(0.75)),
        "cheating_index": float(df["cheating_index"].quantile(0.75)),
        "jerk_score": float(df["jerk_score"].quantile(0.75)),
    }
    return index, q


def _conflict_flags(pred_text: str, stat_row: dict, q: dict) -> List[str]:
    if stat_row is None:
        return []
    t = str(pred_text)
    flags = []
    says_stable = bool(re.search(r"(很稳|稳定|非常标准|教科书|不抖|无抖动)", t))
    says_no_cheat = bool(re.search(r"(不借力|没有借力|很标准)", t))
    says_smooth = bool(re.search(r"(流畅|平稳|节奏好)", t))

    if says_stable and float(stat_row.get("tremor_score", 0.0)) >= q["tremor_score"]:
        flags.append("tremor_conflict")
    if says_no_cheat and float(stat_row.get("cheating_index", 0.0)) >= q["cheating_index"]:
        flags.append("cheating_conflict")
    if says_smooth and float(stat_row.get("jerk_score", 0.0)) >= q["jerk_score"]:
        flags.append("jerk_conflict")
    return flags


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    collapse_versions = not args.no_collapse_versions

    gt_rows = _read_jsonl(args.gt_path)
    pred_rows = _read_jsonl(args.pred_path)

    # 1) 预测映射：可按 rep(base_signal_id) 聚合
    pred_map: Dict[str, dict] = {}
    for r in pred_rows:
        sid = r.get("signal_id")
        if not sid:
            continue
        key = _base_signal_id(sid) if collapse_versions else sid
        pred_text = r.get("pred_text", r.get("text", r.get("output_text", "")))
        # 同 key 多条时保留更长文本（通常信息量更高）
        old = pred_map.get(key, {})
        if (not old) or (len(str(pred_text)) > len(str(old.get("pred_text", "")))):
            pred_map[key] = {"pred_text": pred_text}

    # 2) GT 聚合：默认把 v0/v1/v2 合并为同一 rep
    if collapse_versions:
        agg: Dict[str, dict] = {}
        for g in gt_rows:
            sid = g.get("signal_id")
            if not sid:
                continue
            key = _base_signal_id(sid)
            gt_tags = _normalize_gt_tags(g.get("tags", ""))
            if key not in agg:
                agg[key] = {
                    "signal_id": key,
                    "text": g.get("text", ""),
                    "tags_set": set(gt_tags),
                }
            else:
                agg[key]["tags_set"].update(gt_tags)
                # 保留更长参考文本，减少短句偏置
                if len(str(g.get("text", ""))) > len(str(agg[key]["text"])):
                    agg[key]["text"] = g.get("text", "")
        gt_rows_eval = [
            {"signal_id": v["signal_id"], "text": v["text"], "tags_set": v["tags_set"]}
            for v in agg.values()
        ]
    else:
        gt_rows_eval = []
        for g in gt_rows:
            sid = g.get("signal_id")
            if not sid:
                continue
            gt_rows_eval.append(
                {
                    "signal_id": sid,
                    "text": g.get("text", ""),
                    "tags_set": _normalize_gt_tags(g.get("tags", "")),
                }
            )

    analyse_idx, q = _build_analyse_index(args.analyse_path)

    labels = list(TAG_PATTERNS.keys())
    per_label: Dict[str, Score] = {k: Score() for k in labels}
    micro = Score()
    per_person_rows = []
    badcases = []
    rouge_scores = []
    n_conflict = 0
    n_conflict_eligible = 0
    n_missing_pred = 0
    n_total = 0
    n_missing_pred = 0

    person_bucket: Dict[str, List[dict]] = {}

    for g in gt_rows_eval:
        sid = g.get("signal_id")
        if not sid:
            continue
        n_total += 1
        pred = pred_map.get(sid)
        if pred is None:
            n_missing_pred += 1
            # matched 口径下，缺失预测不计分，只统计覆盖率
            if args.eval_scope == "matched":
                continue

            # all 口径下：缺失预测按漏检处理（FN）
            gt_tags = set(g.get("tags_set", set()))
            for lb in labels:
                if lb in gt_tags:
                    per_label[lb].fn += 1
                    micro.fn += 1
            continue

        gt_tags = set(g.get("tags_set", set()))
        pred_tags = _extract_pred_tags(pred.get("pred_text", ""))

        # 标签统计
        for lb in labels:
            gt_has = lb in gt_tags
            pr_has = lb in pred_tags
            if gt_has and pr_has:
                per_label[lb].tp += 1
                micro.tp += 1
            elif (not gt_has) and pr_has:
                per_label[lb].fp += 1
                micro.fp += 1
            elif gt_has and (not pr_has):
                per_label[lb].fn += 1
                micro.fn += 1

        # 文本相似度
        rouge = rouge_l_f1(pred.get("pred_text", ""), g.get("text", ""))
        rouge_scores.append(rouge)

        # 冲突检测
        conflict_flags = []
        if analyse_idx is not None:
            try:
                key = _parse_signal_id(sid)
                stat_row = analyse_idx.get(key)
                if stat_row is not None:
                    n_conflict_eligible += 1
                    conflict_flags = _conflict_flags(pred.get("pred_text", ""), stat_row, q)
                    if conflict_flags:
                        n_conflict += 1
            except Exception:
                pass

        person_id = sid.split("_W")[0] if "_W" in sid else (sid.split("_C")[0] if "_C" in sid else "UNKNOWN")
        item = {
            "signal_id": sid,
            "person_id": person_id,
            "gt_tags": sorted(list(gt_tags)),
            "pred_tags": sorted(list(pred_tags)),
            "gt_text": g.get("text", ""),
            "pred_text": pred.get("pred_text", ""),
            "rouge_l_f1": rouge,
            "conflict_flags": conflict_flags,
        }
        person_bucket.setdefault(person_id, []).append(item)

        tag_overlap = len(gt_tags & pred_tags) / max(len(gt_tags | pred_tags), 1)
        if (tag_overlap < 0.2) or (len(conflict_flags) > 0):
            badcases.append(item)

    # 汇总指标
    macro_f1 = float(np.mean([per_label[k].f1() for k in labels])) if labels else 0.0
    micro_f1 = micro.f1()
    rouge_mean = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    conflict_rate = (
        float(n_conflict / max(n_conflict_eligible, 1))
        if n_conflict_eligible > 0
        else 0.0
    )

    # 按人统计
    for pid, items in person_bucket.items():
        p_micro = Score()
        for it in items:
            gt_tags = set(it["gt_tags"])
            pr_tags = set(it["pred_tags"])
            for lb in labels:
                gt_has = lb in gt_tags
                pr_has = lb in pr_tags
                if gt_has and pr_has:
                    p_micro.tp += 1
                elif (not gt_has) and pr_has:
                    p_micro.fp += 1
                elif gt_has and (not pr_has):
                    p_micro.fn += 1
        per_person_rows.append(
            {
                "person_id": pid,
                "n_samples": len(items),
                "tag_micro_f1": p_micro.f1(),
                "rouge_l_f1_mean": float(np.mean([x["rouge_l_f1"] for x in items])) if items else 0.0,
                "conflict_rate": float(np.mean([1.0 if x["conflict_flags"] else 0.0 for x in items])) if items else 0.0,
            }
        )

    per_label_f1 = {k: per_label[k].f1() for k in labels}
    summary = {
        "eval_scope": args.eval_scope,
        "collapse_versions": collapse_versions,
        "n_total_gt": n_total,
        "n_pred_matched": n_total - n_missing_pred,
        "n_missing_pred": n_missing_pred,
        "coverage": float((n_total - n_missing_pred) / max(n_total, 1)),
        "tag_micro_f1": micro_f1,
        "tag_macro_f1": macro_f1,
        "per_label_f1": per_label_f1,
        "rouge_l_f1_mean": rouge_mean,
        "conflict_rate": conflict_rate,
        "n_conflict_eligible": n_conflict_eligible,
        "n_conflict": n_conflict,
        "pass_suggestion": {
            "tag_micro_f1_ge_0.60": micro_f1 >= 0.60,
            "conflict_rate_le_0.15": conflict_rate <= 0.15,
        },
    }

    # 保存
    with open(os.path.join(args.out_dir, "phase3_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(per_person_rows).sort_values("person_id").to_csv(
        os.path.join(args.out_dir, "phase3_eval_by_person.csv"),
        index=False,
        encoding="utf-8",
    )

    with open(os.path.join(args.out_dir, "phase3_badcases.jsonl"), "w", encoding="utf-8") as f:
        for r in badcases:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[Phase3 Eval] done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

