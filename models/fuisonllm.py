import torch
import torch.nn as nn

class Phase3FusionLLM(nn.Module):
    def __init__(self, phase1_encoders, phase2_fusion, llm_model, reprogramming_layer):
        super().__init__()
        # 1. 继承并冻结之前的成果 (Phase 1 & 2)
        self.encoders = phase1_encoders
        self.fusion_layer = phase2_fusion
        for param in self.encoders.parameters():
            param.requires_grad = False
        for param in self.fusion_layer.parameters():
            param.requires_grad = False
            
        # 2. 核心：重编程层 (这是 Phase 3 唯一开启梯度的部分)
        self.reprogramming_layer = reprogramming_layer
        
        # 3. LLM 底座 (冻结，建议开启 Gradient Checkpointing 节省显存)
        self.llm = llm_model
        for param in self.llm.parameters():
            param.requires_grad = False
            
    def forward(self, emg, imu, prompt_ids, stat_ids, target_ids=None):
        """
        Args:
            emg, imu: 传感器原始信号
            prompt_ids: 第一层 (System Prompt) 的 Token IDs
            stat_ids: 第二层 (统计指标文本) 的 Token IDs
            target_ids: 目标点评文本 (Coaching Text) 的 Token IDs (训练时使用)
        """
        batch_size = emg.size(0)

        # --- Step A: 信号特征提取 ---
        with torch.no_grad():
            h_emg = self.encoders.emg(emg)
            h_imu = self.encoders.imu(imu)
            h_fused = self.fusion_layer(h_emg, h_imu)

        # --- Step B: 信号语义化 (Soft Tokens) ---
        # 假设 reprogramming 输出 (B, n_soft_tokens, 4096)
        soft_tokens = self.reprogramming_layer(h_fused)
        n_soft = soft_tokens.size(1)

        # --- Step C: 多模态 Embeddings 拼接 ---
        # 1. 提取各部分文本的 Embedding
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids) # (B, S1, 4096)
        stat_embeds = self.llm.get_input_embeddings()(stat_ids)     # (B, S2, 4096)
        
        if target_ids is not None:
            # 训练模式：拼接 [Prompt] + [Stats] + [Signal] + [Target]
            target_embeds = self.llm.get_input_embeddings()(target_ids)
            full_embeds = torch.cat([prompt_embeds, stat_embeds, soft_tokens, target_embeds], dim=1)
        else:
            # 推理模式：拼接 [Prompt] + [Stats] + [Signal]
            full_embeds = torch.cat([prompt_embeds, stat_embeds, soft_tokens], dim=1)

        # --- Step D: 标签与 Mask 处理 (训练核心) ---
        if target_ids is not None:
            # 我们只希望 LLM 对 target_ids 部分计算 Loss
            # 前面的 Prompt, Stats, Soft Tokens 部分全部填 -100
            s1, s2, s3 = prompt_ids.size(1), stat_ids.size(1), n_soft
            target_len = target_ids.size(1)
            
            labels = torch.full((batch_size, s1 + s2 + s3 + target_len), -100).to(full_embeds.device)
            # 只有最后 target 部分需要预测
            labels[:, s1 + s2 + s3:] = target_ids 
        else:
            labels = None

        # --- Step E: 让 LLM 预测 ---
        # 注意：Qwen 等模型需要 attention_mask
        attention_mask = torch.ones(full_embeds.shape[:2], device=full_embeds.device)
        
        outputs = self.llm(
            inputs_embeds=full_embeds,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.loss, outputs.logits