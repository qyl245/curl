import torch
import torch.nn as nn
import torch.nn.functional as F

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, num_prototypes, llm_dim, n_soft_tokens=4):
        """
        Args:
            d_model: Phase 2 融合特征的维度 (例如 128)
            num_prototypes: 语义原型的数量 (例如 64)
            llm_dim: LLM 的隐藏层维度 (Qwen-7B 是 4096)
            n_soft_tokens: 每个动作信号转化成的 Soft Tokens 数量 (建议 4-8 个)
        """
        super().__init__()
        self.d_model = d_model
        self.llm_dim = llm_dim
        self.n_soft_tokens = n_soft_tokens
        
        # 1. 可学习的文本原型 (Prototypes)，保持在 d_model 空间
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        
        # 1b. extract.py 产出 4096 维时，投影到 d_model 再初始化 (方案B：更省算力)
        self.init_proj = nn.Linear(llm_dim, d_model)
        
        # 2. 增强稳定性：输入归一化
        self.input_norm = nn.LayerNorm(d_model)
        
        # 3. 跨模态注意力：将物理信号映射到原型空间
        self.query_embed = nn.Embedding(n_soft_tokens, d_model)
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True, dropout=0.1)
        
        # 4. 投影层：从 d_model 映射到 llm_dim
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, llm_dim),
            nn.LayerNorm(llm_dim)
        )

    def init_with_tokens(self, token_embeddings):
        """
        用 Qwen 词向量初始化原型。
        token_embeddings: (num_keywords, llm_dim) 或 (num_keywords, d_model)
        extract.py 产出 4096 维时，经 init_proj 投影到 d_model 再写入。
        """
        with torch.no_grad():
            inp_dim = token_embeddings.size(-1)
            if inp_dim == self.llm_dim:
                # 4096 维：投影到 d_model
                projected = self.init_proj(token_embeddings)
                assert projected.size(-1) == self.d_model, f"投影后维度应为 {self.d_model}，得到 {projected.size(-1)}"
            elif inp_dim == self.d_model:
                projected = token_embeddings
            else:
                raise ValueError(f"token_embeddings 最后一维应为 {self.llm_dim} 或 {self.d_model}，得到 {inp_dim}")
            
            num_keywords = projected.size(0)
            n_write = min(num_keywords, self.prototypes.size(0))
            self.prototypes[:n_write].copy_(projected[:n_write])
            if num_keywords > n_write:
                print(f"[ReprogrammingLayer] 原型输入数 {num_keywords} 超过上限 {n_write}，已截断写入。")
            print(f"[ReprogrammingLayer] init_with_tokens: 输入 {token_embeddings.shape} -> 原型 {n_write} 个已初始化，维度一致 d_model={self.d_model}")

    def forward(self, h_fused):
        """
        h_fused: 来自 Phase 2 的融合特征, 形状 (B, d_model)
        """
        h_fused = self.input_norm(h_fused)
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