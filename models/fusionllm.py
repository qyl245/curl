import torch
import torch.nn as nn

class Phase3FusionLLM(nn.Module):
    def __init__(self, phase2_model, llm_model, reprogramming_layer):
        super().__init__()
        # 1. 复用 Phase 2 主模型，直接取 fused_embedding，不再手写 encoders + fusion_layer
        self.phase2_model = phase2_model
        for param in self.phase2_model.parameters():
            param.requires_grad = False
        self.phase2_model.eval()
            
        # 2. 重编程层 (Phase 3 唯一可训练部分)
        self.reprogramming_layer = reprogramming_layer
        
        # 3. LLM 底座 (冻结)
        self.llm = llm_model
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

    def train(self, mode: bool = True):
        """
        仅训练重编程层；冻结的 Phase2/LLM 固定为 eval，避免 dropout 造成特征漂移。
        """
        super().train(mode)
        self.phase2_model.eval()
        self.llm.eval()
        self.reprogramming_layer.train(mode)
        return self
            
    def forward(
        self,
        emg,
        imu,
        prompt_ids,
        stat_ids,
        target_ids=None,
        prompt_attention_mask=None,
        stat_attention_mask=None,
        target_attention_mask=None,
        pad_token_id=None,
    ):
        """
        Args:
            emg, imu: 传感器原始信号
            prompt_ids, stat_ids, target_ids: 各段 Token IDs
            prompt_attention_mask, stat_attention_mask, target_attention_mask: 对应 attention_mask，padding 位为 0
            pad_token_id: target 中 pad 位置在 labels 中置为 -100，避免监督污染
        """
        batch_size = emg.size(0)

        # --- Step A: 复用 Phase 2 前向，取 fused_embedding ---
        with torch.no_grad():
            out = self.phase2_model(emg, imu)
            h_fused = out["fused_embedding"]  # (B, d_model)

        # --- Step B: 信号语义化 (Soft Tokens) ---
        soft_tokens = self.reprogramming_layer(h_fused)
        n_soft = soft_tokens.size(1)

        # --- Step C: 多模态 Embeddings 拼接 ---
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        stat_embeds = self.llm.get_input_embeddings()(stat_ids)

        if target_ids is not None:
            target_embeds = self.llm.get_input_embeddings()(target_ids)
            full_embeds = torch.cat([prompt_embeds, stat_embeds, soft_tokens, target_embeds], dim=1)
        else:
            full_embeds = torch.cat([prompt_embeds, stat_embeds, soft_tokens], dim=1)
        # 对齐到 LLM 权重 dtype，避免 Float/BFloat16 matmul 冲突（常见于验证阶段）
        llm_dtype = self.llm.get_input_embeddings().weight.dtype
        full_embeds = full_embeds.to(dtype=llm_dtype)

        # --- Step D: 标签与 Mask 处理 ---
        s1, s2 = prompt_ids.size(1), stat_ids.size(1)
        seq_len = full_embeds.size(1)

        # D1. labels: Prompt/Stats/SoftTokens 填 -100；Target 段 padding 位置填 -100
        if target_ids is not None:
            labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=full_embeds.device)
            target_slice = labels[:, s1 + s2 + n_soft :]
            if target_attention_mask is not None:
                target_slice.copy_(
                    torch.where(
                        target_attention_mask.to(dtype=torch.bool),
                        target_ids,
                        torch.full_like(target_ids, -100),
                    )
                )
            elif pad_token_id is not None:
                target_slice.copy_(
                    torch.where(target_ids == pad_token_id, torch.full_like(target_ids, -100), target_ids)
                )
            else:
                target_slice.copy_(target_ids)
        else:
            labels = None

        # D2. attention_mask: 拼接 prompt/stat/target 的 mask，soft token 段全 1
        if (
            prompt_attention_mask is not None
            and stat_attention_mask is not None
            and target_attention_mask is not None
        ):
            soft_mask = torch.ones(batch_size, n_soft, device=full_embeds.device, dtype=prompt_attention_mask.dtype)
            attention_mask = torch.cat(
                [prompt_attention_mask, stat_attention_mask, soft_mask, target_attention_mask], dim=1
            )
            assert attention_mask.size(1) == seq_len, f"mask 长度 {attention_mask.size(1)} 与 full_embeds {seq_len} 不一致"
        else:
            attention_mask = torch.ones(batch_size, seq_len, device=full_embeds.device)

        outputs = self.llm(
            inputs_embeds=full_embeds,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.loss, outputs.logits
