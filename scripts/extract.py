"""
从 Qwen 词向量提取语义原型，用于 Phase3 ReprogrammingLayer 初始化。
使用 PCA 将 4096 维投影到 d_model，保留语义主成分，避免随机投影削弱语义。
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config

MANUAL_KEYWORDS = [
    "标准", "教科书", "借力", "晃动", "挣扎", "抖动", "费劲", "核心", "控制",
]


def extract_prototypes(
    model_path: str,
    jsonl_path: str,
    save_path: str,
    num_prototypes: int = 64,
    d_model: int = 128,
):
    """
    Args:
        model_path: Qwen 模型路径 (HuggingFace 或本地)
        jsonl_path: coaching.jsonl 路径
        save_path: 输出 prototypes_init.pt 路径
        num_prototypes: 原型数量
        d_model: 目标维度，与 Phase2 fusion embed_dim 一致，避免随机投影
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", trust_remote_code=True
    )
    embeddings = model.get_input_embeddings().weight.data  # (vocab_size, 4096), 可能是 bf16

    def _to_numpy_fp32(x: torch.Tensor) -> np.ndarray:
        # numpy 不支持直接从 bf16 张量转换，统一先转 float32
        return x.detach().float().cpu().numpy()

    # 1. 收集标签
    all_tags = list(MANUAL_KEYWORDS)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            for t in data.get("tags", "").split("/"):
                t = t.strip()
                if t and t not in all_tags:
                    all_tags.append(t)
    print(f"提取了 {len(all_tags)} 个唯一标签")

    # 2. 计算 tag embedding，处理空 token 边界
    tag_vectors = []
    emb_mean = _to_numpy_fp32(embeddings.mean(dim=0))
    for tag in all_tags:
        token_ids = tokenizer.encode(tag, add_special_tokens=False)
        if len(token_ids) == 0:
            tag_vec = emb_mean.copy()
        else:
            tid_t = torch.tensor(token_ids, device=embeddings.device, dtype=torch.long)
            tag_vec = _to_numpy_fp32(embeddings[tid_t].mean(dim=0))
        tag_vectors.append(tag_vec)
    tag_vectors = np.array(tag_vectors, dtype=np.float32)  # (n_tags, 4096)

    # 3. PCA 投影到 d_model，保留语义主成分（避免随机投影削弱语义）
    n_comp = min(d_model, tag_vectors.shape[0], tag_vectors.shape[1])
    if n_comp < d_model:
        print(f"标签/维度不足，PCA 使用 n_components={n_comp}")
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(tag_vectors)  # (n_tags, n_comp)
    if reduced.shape[1] < d_model:
        pad = np.zeros((reduced.shape[0], d_model - reduced.shape[1]), dtype=np.float32)
        reduced = np.hstack([reduced, pad])
    print(f"PCA 完成，保留方差比: {pca.explained_variance_ratio_.sum():.4f}")

    # 4. K-Means 聚类到 num_prototypes
    if len(reduced) > num_prototypes:
        kmeans = KMeans(n_clusters=num_prototypes, n_init=10, random_state=42)
        prototype_centers = kmeans.fit(reduced).cluster_centers_
    else:
        # 标签不足时，已有向量 + 小噪声填充
        padding = np.random.randn(num_prototypes - len(reduced), d_model).astype(np.float32) * 0.01
        prototype_centers = np.vstack([reduced, padding])

    # 5. 保存 (num_prototypes, d_model)，init_with_tokens 可直接写入
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(torch.tensor(prototype_centers, dtype=torch.float32), save_path)
    print(f"原型已保存: {save_path}, shape={prototype_centers.shape}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="提取 Phase3 语义原型")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_path", type=str, default=None, help="覆盖 config 中的 llm_path")
    parser.add_argument("--jsonl", type=str, default=None, help="覆盖 coaching.jsonl 路径")
    args = parser.parse_args()

    config = load_config(args.config)
    p3 = config.get("phase3", {})
    data_p3 = config.get("data", {}).get("phase3", {})
    paths = config.get("paths", {})

    model_path = args.model_path or p3.get("llm_path", "Qwen/Qwen-7B")
    jsonl_path = args.jsonl or data_p3.get("coaching_path", "dataset/coaching.jsonl")
    save_path = paths.get("prototypes_init", "dataset/prototypes_init.pt")
    num_prototypes = p3.get("num_prototypes", 64)
    d_model = config.get("model", {}).get("fusion", {}).get("embed_dim", 128)

    extract_prototypes(
        model_path=model_path,
        jsonl_path=jsonl_path,
        save_path=save_path,
        num_prototypes=num_prototypes,
        d_model=d_model,
    )


if __name__ == "__main__":
    main()
