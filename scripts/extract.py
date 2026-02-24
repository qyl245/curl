import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np

def extract_prototypes(model_path, jsonl_path, num_prototypes=64):
    # 1. 加载模型（在 5090 上可以快进加载）
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True)
    embeddings = model.get_input_embeddings().weight.data # 拿到 4096 维的矩阵
    
    # 2. 获取人工定义的关键词 (你的图1逻辑)
    manual_keywords = ["标准", "教科书", "借力", "晃动", "挣扎", "抖动", "费劲", "核心", "控制"]
    
    # 3. 从 JSONL 提取所有独特的 tags
    all_tags = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 这里的 tags 可能是 "动作流畅度下降 / 爆发式借力"
            split_tags = [t.strip() for t in data['tags'].split('/')]
            all_tags.extend(split_tags)
    
    unique_tags = list(set(all_tags + manual_keywords))
    print(f"提取了 {len(unique_tags)} 个唯一标签")

    # 4. 计算这些标签的 Embedding
    tag_vectors = []
    for tag in unique_tags:
        token_ids = tokenizer.encode(tag, add_special_tokens=False)
        # 在 Embedding 矩阵中查表并取平均
        tag_vec = embeddings[token_ids].mean(dim=0).cpu().numpy()
        tag_vectors.append(tag_vec)
    
    tag_vectors = np.array(tag_vectors)

    # 5. 使用 K-Means 聚类到指定的原型数量
    # 这样即使你有 100 个标签，也会浓缩成 64 个最具代表性的“语义中心”
    if len(tag_vectors) > num_prototypes:
        kmeans = KMeans(n_clusters=num_prototypes, n_init=10)
        prototype_centers = kmeans.fit(tag_vectors).cluster_centers_
    else:
        # 如果标签不够，就用随机噪声填充剩余部分
        padding = np.random.randn(num_prototypes - len(tag_vectors), 4096) * 0.01
        prototype_centers = np.vstack([tag_vectors, padding])

    # 6. 保存结果供 Phase 3 使用
    torch.save(torch.tensor(prototype_centers), "dataset/prototypes_init.pt")
    print(f"原型初始化权重已保存，维度: {prototype_centers.shape}")

if __name__ == "__main__":
    extract_prototypes("Qwen-7B", "dataset/coaching.jsonl")