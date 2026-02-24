import torch
import torch.nn as nn
import torch.nn.functional as F

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, num_prototypes, llm_dim, n_soft_tokens=4):
        """
        Args:
            d_model: Phase 2 融合特征的维度 (例如 512)
            num_prototypes: 语义原型的数量 (例如 64)
            llm_dim: LLM 的隐藏层维度 (Qwen-7B 是 4096)
            n_soft_tokens: 每个动作信号转化成的 Soft Tokens 数量 (建议 4-8 个)
        """
        super().__init__()
        self.d_model = d_model
        self.n_soft_tokens = n_soft_tokens
        
        # 1. 可学习的文本原型 (Prototypes)
        # 我们用 Parameter 定义，方便后续用 Qwen 的 Embedding 初始化
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        
        # 2. 增强稳定性：输入归一化
        self.input_norm = nn.LayerNorm(d_model)
        
        # 3. 跨模态注意力：将物理信号映射到原型空间
        # 我们增加一个可学习的 Query (n_soft_tokens 个)，让模型决定提取哪几个维度的语义
        self.query_embed = nn.Embedding(n_soft_tokens, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead=8, batch_first=True, dropout=0.1)
        
        # 4. 投影层：从 d_model (512) 映射到 llm_dim (4096)
        # 使用两层 MLP 增加表达能力
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, llm_dim),
            nn.LayerNorm(llm_dim) # 确保输出给 LLM 的向量分布稳定
        )

    def init_with_tokens(self, token_embeddings):
        """
        核心完善：用 Qwen 的词向量初始化原型
        token_embeddings: 形状为 (num_actual_keywords, d_model) 的张量
        """
        with torch.no_grad():
            num_keywords = token_embeddings.size(0)
            if num_keywords <= self.prototypes.size(0):
                # 用实际词向量覆盖前一部分
                self.prototypes[:num_keywords].copy_(token_embeddings)
                # 剩余部分保持随机或加入微小扰动
            print(f"成功使用 {num_keywords} 个核心词向量初始化原型。")

    def forward(self, h_fused):
        """
        h_fused: 来自 Phase 2 的融合特征, 形状 (B, d_model)
        """
        batch_size = h_fused.size(0)
        
        # 1. 准备 Query: (B, n_soft_tokens, d_model)
        # 我们不直接用 h_fused 做 Query，而是用它来指导 Query
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = queries + h_fused.unsqueeze(1) # 注入当前动作的特征
        
        # 2. 准备 Key/Value (Prototypes): (B, num_prototypes, d_model)
        keys = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. 跨模态注意力计算
        # 模型会根据当前的动作特征，去从“原型字典”里查表
        reprogrammed_feat, _ = self.attn(
            query=queries, 
            key=keys, 
            value=keys
        )
        
        # 4. 映射到 LLM 空间
        # 输出形状: (B, n_soft_tokens, 4096)
        llm_tokens = self.proj(reprogrammed_feat)
        
        return llm_tokens