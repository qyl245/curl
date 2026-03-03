import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_cosine_schedule_with_warmup, AutoTokenizer
import os
from tqdm import tqdm
from utils.logging_utils import setup_logger

logger = setup_logger()

def _cfg_get(config, key, default=None):
    """兼容 dict 与 object 的 config，优先从 phase3 子配置读取。"""
    if isinstance(config, dict):
        return config.get("phase3", {}).get(key, config.get(key, default))
    return getattr(config, key, default)


class Phase3Trainer:
    def __init__(self, model, train_dataset, val_dataset, config, device):
        self.config = config
        self.device = device
        self.model = model.to(device)

        llm_path = _cfg_get(config, "llm_path")
        if not llm_path:
            raise ValueError("config 中需指定 phase3.llm_path")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        # 某些 Qwen tokenizer 不支持 add_special_tokens，这里仅选取可用 id 做手工 padding。
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.unk_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0
        logger.info(
            f"Phase3 tokenizer: pad_token={self.tokenizer.pad_token}, "
            f"pad_token_id={self.pad_token_id}, eos_token={self.tokenizer.eos_token}"
        )

        bs = _cfg_get(config, "batch_size_p3", 4)
        epochs = _cfg_get(config, "epochs_p3", 5)
        lr = _cfg_get(config, "lr_p3", 1e-4)
        ckpt_dir = _cfg_get(config, "checkpoint_dir") or "outputs/models"

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            collate_fn=self.collate_fn,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=len(self.train_loader) * 2,
            num_training_steps=len(self.train_loader) * epochs,
        )
        self._epochs_p3 = epochs
        self._checkpoint_dir = ckpt_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def _tokenize_and_pad(self, texts):
        """
        不依赖 tokenizer 内置 padding，避免 Qwen 不支持 pad token 的报错。
        """
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        ids_list = [torch.tensor(x, dtype=torch.long) for x in encoded["input_ids"]]
        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).long()
        return input_ids, attention_mask

    def collate_fn(self, batch):
        """
        核心：将 batch 中的不同长度文本转为对齐的 ID，并返回各段 attention_mask
        """
        emgs = torch.stack([item['emg'] for item in batch])
        imus = torch.stack([item['imu'] for item in batch])

        system_prompt = "你是一位专业的健身教练，请根据以下运动数据和信号，给出一句精准的动作点评。"

        prompt_ids, prompt_mask = self._tokenize_and_pad([system_prompt] * len(batch))
        stat_ids, stat_mask = self._tokenize_and_pad([item['stat_text'] for item in batch])
        target_ids, target_mask = self._tokenize_and_pad([item['target_text'] for item in batch])

        return {
            'emg': emgs.to(self.device),
            'imu': imus.to(self.device),
            'prompt_ids': prompt_ids.to(self.device),
            'stat_ids': stat_ids.to(self.device),
            'target_ids': target_ids.to(self.device),
            'prompt_attention_mask': prompt_mask.to(self.device),
            'stat_attention_mask': stat_mask.to(self.device),
            'target_attention_mask': target_mask.to(self.device),
        }

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        use_amp = str(self.device).startswith("cuda")
        
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            
            # 使用 BF16 混合精度 (针对 5090 优化)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = self.model(
                    emg=batch['emg'],
                    imu=batch['imu'],
                    prompt_ids=batch['prompt_ids'],
                    stat_ids=batch['stat_ids'],
                    target_ids=batch['target_ids'],
                    prompt_attention_mask=batch['prompt_attention_mask'],
                    stat_attention_mask=batch['stat_attention_mask'],
                    target_attention_mask=batch['target_attention_mask'],
                    pad_token_id=self.pad_token_id,
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
        use_amp = str(self.device).startswith("cuda")
        for batch in tqdm(self.val_loader, desc="Validating"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = self.model(
                    emg=batch['emg'],
                    imu=batch['imu'],
                    prompt_ids=batch['prompt_ids'],
                    stat_ids=batch['stat_ids'],
                    target_ids=batch['target_ids'],
                    prompt_attention_mask=batch['prompt_attention_mask'],
                    stat_attention_mask=batch['stat_attention_mask'],
                    target_attention_mask=batch['target_attention_mask'],
                    pad_token_id=self.pad_token_id,
                )
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss):
        save_path = os.path.join(self._checkpoint_dir, f"phase3_repro_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.reprogramming_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, save_path)
        logger.info(f"Saved Phase 3 weights to {save_path}")

    def train(self):
        best_loss = float('inf')
        for epoch in range(self._epochs_p3):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)