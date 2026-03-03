"""
Phase3 推理脚本：生成 outputs/phase3_preds.jsonl 供 eval_phase3.py 使用。

默认流程：
1) 加载 Phase2 编码与融合权重
2) 构建 Phase3 模型并加载 ReprogrammingLayer checkpoint
3) 在 Phase3Dataset 的 split(train/val) 上生成点评文本
4) 输出 JSONL: {"signal_id", "pred_text", "model_ckpt", "person_id"}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import re
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 确保从 scripts/ 启动时也能导入项目内模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.create_dataset import get_num_phase2_classes
from dataloaders.phase3_dataset import Phase3Dataset
from main import _load_ckpt_if_exists, build_encoder
from models.fusion import MultiModalModel
from models.fusionllm import Phase3FusionLLM
from models.reprogramming import ReprogrammingLayer
from utils.config import load_config
from utils.logging_utils import setup_logger

logger = setup_logger()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize_and_pad(tokenizer, texts: List[str], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        return_attention_mask=False,
        padding=False,
        truncation=True,
    )
    ids_list = [torch.tensor(x, dtype=torch.long) for x in encoded["input_ids"]]
    input_ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()
    return input_ids, attention_mask


def _resolve_repro_ckpt(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        return path_or_dir
    cand = sorted(glob.glob(os.path.join(path_or_dir, "phase3_repro_epoch_*.pt")))
    if not cand:
        raise FileNotFoundError(f"未找到 phase3_repro checkpoint: {path_or_dir}")
    return cand[-1]


def _parse_person_from_signal_id(signal_id: str) -> str:
    if "_W" in signal_id:
        return signal_id.split("_W")[0]
    return "UNKNOWN"


def _base_signal_id(signal_id: str) -> str:
    return re.sub(r"_v\d+$", "", signal_id)


def _resolve_arg(cli_value, cfg_value):
    """命令行优先；若未传则回落到配置值。"""
    return cfg_value if cli_value is None else cli_value


@torch.no_grad()
def generate_text_batch(
    model: Phase3FusionLLM,
    tokenizer,
    device: str,
    emg: torch.Tensor,
    imu: torch.Tensor,
    stat_texts: List[str],
    max_new_tokens: int,
    min_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[str]:
    batch_size = emg.size(0)
    system_prompt = "你是一位专业的健身教练，请根据以下运动数据和信号，给出一句精准的动作点评。"
    prompts = [system_prompt] * batch_size

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.unk_token_id
    if pad_token_id is None:
        pad_token_id = 0

    prompt_ids, prompt_mask = _tokenize_and_pad(tokenizer, prompts, pad_token_id)
    stat_ids, stat_mask = _tokenize_and_pad(tokenizer, stat_texts, pad_token_id)

    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    stat_ids = stat_ids.to(device)
    stat_mask = stat_mask.to(device)
    emg = emg.to(device)
    imu = imu.to(device)

    # 复用 fusionllm 的前半流程，构造 inputs_embeds
    out = model.phase2_model(emg, imu)
    h_fused = out["fused_embedding"]
    soft_tokens = model.reprogramming_layer(h_fused)
    n_soft = soft_tokens.size(1)

    prompt_embeds = model.llm.get_input_embeddings()(prompt_ids)
    stat_embeds = model.llm.get_input_embeddings()(stat_ids)
    full_embeds = torch.cat([prompt_embeds, stat_embeds, soft_tokens], dim=1)
    full_embeds = full_embeds.to(dtype=model.llm.get_input_embeddings().weight.dtype)

    soft_mask = torch.ones(batch_size, n_soft, device=device, dtype=prompt_mask.dtype)
    attention_mask = torch.cat([prompt_mask, stat_mask, soft_mask], dim=1)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        # 某些模型在仅传 inputs_embeds 时要求显式提供 bos_token_id
        "bos_token_id": (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else (pad_token_id if pad_token_id is not None else 0)
        ),
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    use_amp = str(device).startswith("cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        gen_ids = model.llm.generate(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # 某些模型在 inputs_embeds 模式下只返回新生成 token，另一些会返回全长序列。
    # 做鲁棒切片，避免把输出切成空串。
    prefix_len = full_embeds.size(1)
    if gen_ids.size(1) > prefix_len:
        pred_ids = gen_ids[:, prefix_len:]
    else:
        pred_ids = gen_ids
    texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def main():
    parser = argparse.ArgumentParser(description="Run phase3 inference and dump JSONL predictions.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val"])
    parser.add_argument("--repro_ckpt", type=str, default=None, help="phase3_repro_epoch_*.pt 文件或目录")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--dedup_by_rep", action="store_true", help="按去掉 v 后的 rep key 去重，只生成一条预测。")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    p3 = config.get("phase3", {})
    infer_cfg = p3.get("infer", {})
    gen_cfg = infer_cfg.get("generation", {})
    train_cfg = config.get("train", {})

    split = _resolve_arg(args.split, infer_cfg.get("split", "val"))
    output_path = _resolve_arg(args.output, infer_cfg.get("output", "outputs/phase3_preds.jsonl"))
    infer_bs = int(_resolve_arg(args.batch_size, infer_cfg.get("batch_size", 8)))
    seed = int(_resolve_arg(args.seed, train_cfg.get("seed", 42)))
    max_new_tokens = int(_resolve_arg(args.max_new_tokens, gen_cfg.get("max_new_tokens", 64)))
    min_new_tokens = int(_resolve_arg(args.min_new_tokens, gen_cfg.get("min_new_tokens", 8)))
    do_sample = bool(args.do_sample or bool(gen_cfg.get("do_sample", False)))
    temperature = float(_resolve_arg(args.temperature, gen_cfg.get("temperature", 0.7)))
    top_p = float(_resolve_arg(args.top_p, gen_cfg.get("top_p", 0.9)))
    repetition_penalty = float(_resolve_arg(args.repetition_penalty, gen_cfg.get("repetition_penalty", 1.15)))
    no_repeat_ngram_size = int(_resolve_arg(args.no_repeat_ngram_size, gen_cfg.get("no_repeat_ngram_size", 3)))
    dedup_by_rep = bool(args.dedup_by_rep or bool(infer_cfg.get("dedup_by_rep", False)))

    set_seed(seed)

    paths = config.get("paths", {})
    data_p3 = config.get("data", {}).get("phase3", {})

    device = config.get("train", {}).get("device", "cuda")
    llm_path = p3.get("llm_path")
    if not llm_path:
        raise ValueError("config.phase3.llm_path 未设置。")

    # --- Build Phase2 backbone ---
    num_classes = get_num_phase2_classes(config)
    emg_encoder = build_encoder(config, "emg")
    imu_encoder = build_encoder(config, "imu")
    _load_ckpt_if_exists(emg_encoder, paths.get("phase1_emg_ckpt", "outputs/models/emg_encoder_best.pt"), "EMG encoder")
    _load_ckpt_if_exists(imu_encoder, paths.get("phase1_imu_ckpt", "outputs/models/imu_encoder_best.pt"), "IMU encoder")
    phase2_model = MultiModalModel(
        emg_encoder=emg_encoder,
        imu_encoder=imu_encoder,
        model_cfg=config["model"],
        num_classes=num_classes,
    )
    phase2_ckpt = paths.get("phase2_fusion_ckpt", "outputs/models/best_fusion_model.pt")
    _load_ckpt_if_exists(phase2_model, phase2_ckpt, "Phase2 fusion")

    # --- Reprogramming ---
    d_model = config["model"]["fusion"]["embed_dim"]
    reprogramming_layer = ReprogrammingLayer(
        d_model=d_model,
        num_prototypes=p3.get("num_prototypes", 64),
        llm_dim=p3.get("llm_dim", 4096),
        n_soft_tokens=p3.get("n_soft_tokens", 4),
    )

    ckpt_hint = args.repro_ckpt or p3.get("checkpoint_dir", "outputs/models")
    repro_ckpt = _resolve_repro_ckpt(ckpt_hint)
    ckpt_obj = torch.load(repro_ckpt, map_location="cpu")
    state = ckpt_obj.get("model_state_dict", ckpt_obj)
    reprogramming_layer.load_state_dict(state, strict=True)
    logger.info(f"Loaded Phase3 reprogramming ckpt: {repro_ckpt}")

    # --- LLM ---
    llm_dtype = torch.bfloat16 if (str(device).startswith("cuda") and torch.cuda.is_available()) else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=llm_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

    model = Phase3FusionLLM(
        phase2_model=phase2_model,
        llm_model=llm,
        reprogramming_layer=reprogramming_layer,
    ).to(device)
    model.eval()

    # --- Dataset ---
    ds = Phase3Dataset(
        coaching_path=data_p3.get("coaching_path", "dataset/coaching.jsonl"),
        jtom_pt_path=config["data"]["phase2"]["processed_pt_path"],
        analyse_jtom_path=data_p3.get("analyse_jtom_path", "dataset/analyse_jtom.csv"),
        val_ratio=float(data_p3.get("val_ratio", 0.1)),
        seed=int(config.get("train", {}).get("seed", 42)),
        split=split,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n = len(ds)
    all_indices = list(range(n))
    if dedup_by_rep:
        seen = set()
        uniq = []
        for i in all_indices:
            sid = ds.rows[i]["signal_id"]
            key = _base_signal_id(sid)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(i)
        indices = uniq
    else:
        indices = all_indices
    n_eff = len(indices)
    logger.info(
        "Infer config: "
        f"split={split}, samples={n}, effective_samples={n_eff}, batch_size={infer_bs}, "
        f"max_new={max_new_tokens}, min_new={min_new_tokens}, do_sample={do_sample}, "
        f"rep_penalty={repetition_penalty}, no_repeat_ngram={no_repeat_ngram_size}, dedup_by_rep={dedup_by_rep}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, n_eff, infer_bs), desc="Phase3 Inference"):
            end = min(start + infer_bs, n_eff)
            batch_idx = indices[start:end]
            items = [ds[i] for i in batch_idx]
            rows = [ds.rows[i] for i in batch_idx]

            emg = torch.stack([x["emg"] for x in items], dim=0)
            imu = torch.stack([x["imu"] for x in items], dim=0)
            stat_texts = [x["stat_text"] for x in items]

            preds = generate_text_batch(
                model=model,
                tokenizer=tokenizer,
                device=device,
                emg=emg,
                imu=imu,
                stat_texts=stat_texts,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            for r, pred_text in zip(rows, preds):
                sid = r["signal_id"]
                out = {
                    "signal_id": sid,
                    "base_signal_id": _base_signal_id(sid),
                    "pred_text": pred_text,
                    "model_ckpt": os.path.basename(repro_ckpt),
                    "person_id": _parse_person_from_signal_id(sid),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

