import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup, AutoTokenizer
import os
from tqdm import tqdm
from utils.logging_utils import logger # 假设你已有的日志工具

class Phase3Trainer:
    def __init__(self, model, train_dataset, val_dataset, config, device):
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_path, trust_remote_code=True)
        # Qwen 必须设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1. 准备 DataLoader (需要自定义 collate_fn 来处理文本序列)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size_p3, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size_p3, 
            collate_fn=self.collate_fn
        )

        # 2. 仅对可学习参数进行优化 (即 ReprogrammingLayer)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.lr_p3, weight_decay=0.01)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=len(self.train_loader) * 2, 
            num_training_steps=len(self.train_loader) * config.epochs_p3
        )

    def collate_fn(self, batch):
        """
        核心：将 batch 中的不同长度文本转为对齐的 ID
        """
        emgs = torch.stack([item['emg'] for item in batch])
        imus = torch.stack([item['imu'] for item in batch])
        
        # 定义 System Prompt (你可以根据需求修改)
        system_prompt = "你是一位专业的健身教练，请根据以下运动数据和信号，给出一句精准的动作点评。"
        
        # Tokenize 各个部分
        # 1. System Prompt
        prompts = self.tokenizer([system_prompt] * len(batch), return_tensors="pt", padding=True)
        # 2. 第二层：统计指标文本
        stats = self.tokenizer([item['stat_text'] for item in batch], return_tensors="pt", padding=True)
        # 3. 目标文本：教练点评
        targets = self.tokenizer([item['target_text'] for item in batch], return_tensors="pt", padding=True)

        return {
            'emg': emgs.to(self.device),
            'imu': imus.to(self.device),
            'prompt_ids': prompts['input_ids'].to(self.device),
            'stat_ids': stats['input_ids'].to(self.device),
            'target_ids': targets['input_ids'].to(self.device)
        }

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # 使用 BF16 混合精度 (针对 5090 优化)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _ = self.model(
                    emg=batch['emg'],
                    imu=batch['imu'],
                    prompt_ids=batch['prompt_ids'],
                    stat_ids=batch['stat_ids'],
                    target_ids=batch['target_ids']
                )
            
            loss.backward()
            # 梯度裁剪，防止 LLM 训练早期不稳定
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        for batch in tqdm(self.val_loader, desc="Validating"):
            loss, _ = self.model(
                emg=batch['emg'],
                imu=batch['imu'],
                prompt_ids=batch['prompt_ids'],
                stat_ids=batch['stat_ids'],
                target_ids=batch['target_ids']
            )
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss):
        # 注意：不要保存整个 7B 模型！只保存 ReprogrammingLayer 的权重
        save_path = os.path.join(self.config.checkpoint_dir, f"phase3_repro_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.reprogramming_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, save_path)
        logger.info(f"Saved Phase 3 weights to {save_path}")

    def train(self):
        best_loss = float('inf')
        for epoch in range(self.config.epochs_p3):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)