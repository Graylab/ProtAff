import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from src.models import build_model 

class PairAffinityModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Container 
        self.base_model = build_model(cfg)
        
        # 2. Setup PEFT (LoRA)
        modules_to_save = ["projector", "norm_input", "head_score"]
        
        if cfg.model.lora.modules_to_save:
             extra = OmegaConf.to_container(cfg.model.lora.modules_to_save)
             for m in extra:
                 if m not in modules_to_save:
                     modules_to_save.append(m)

        target_modules = OmegaConf.to_container(cfg.model.lora.target_modules)
        
        peft_config = LoraConfig(
            r=cfg.model.lora.r, 
            lora_alpha=cfg.model.lora.alpha, 
            target_modules=target_modules,
            modules_to_save=modules_to_save, 
            lora_dropout=cfg.model.lora.dropout, 
            bias="none"
        )

        self.model = get_peft_model(self.base_model, peft_config)
        
        # 3. Loss Function (Ranking Only)
        # MarginRankingLoss with y=-1 optimizes for input1 < input2
        # This matches Log Kd logic (Better Binder = Lower Value)
        self.margin = getattr(cfg.training, "margin", 0.1)
        self.rank_loss = nn.MarginRankingLoss(margin=self.margin)
        
        # 4. Transfer Learning
        ckpt_path = cfg.get("pretrained_ckpt_path", None) 
        if ckpt_path:
            self._load_phase1_weights(ckpt_path)

    def forward(self, input_ids, attention_mask):
        """
        Standard forward pass. Returns predicted Log Kd (Lower is better).
        """
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

    def training_step(self, batch, batch_idx):
        # 1. Forward Pass
        # scores_better: Should be LOWER (Stronger binding)
        scores_better = self(
            input_ids=batch['better_input_ids'], 
            attention_mask=batch['better_mask']
        )
        
        # scores_worse: Should be HIGHER (Weaker binding)
        scores_worse = self(
            input_ids=batch['worse_input_ids'], 
            attention_mask=batch['worse_mask']
        )
        
        # 2. Ranking Loss
        # Target is -1.0 implies: Minimize scores_better - scores_worse
        # i.e., Make scores_better smaller than scores_worse
        target_rank = torch.full_like(scores_better, -1.0)
        loss = self.rank_loss(scores_better, scores_worse, target_rank)

        # --- LOGGING ---
        # Accuracy: Percentage where Stronger Binder has Lower Kd
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        scores_better = self(
            input_ids=batch['better_input_ids'], 
            attention_mask=batch['better_mask']
        )
        scores_worse = self(
            input_ids=batch['worse_input_ids'], 
            attention_mask=batch['worse_mask']
        )
        
        # Target -1.0 -> Correct direction for Log Kd
        target_rank = torch.full_like(scores_better, -1.0)
        loss = self.rank_loss(scores_better, scores_worse, target_rank)
        
        # Accuracy: Check if predicted Kd(Better) < predicted Kd(Worse)
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.training.learning_rate, 
            weight_decay=self.cfg.training.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_ratio = getattr(self.cfg.training, "warmup_ratio", 0.1)
        num_warmup_steps = int(total_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _load_phase1_weights(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Path not found: {ckpt_path}. Training from scratch.")
            return

        print(f"\n[INFO] Loading Phase 1 weights from {ckpt_path}")
        try:
            raw_weights = load_peft_weights(ckpt_path, device="cpu")
            current_keys = list(self.model.state_dict().keys())
            
            def normalize(k):
                ignore = ['base_model', 'model', 'module', 'modules_to_save', 'default']
                parts = k.split('.')
                return '.'.join([p for p in parts if p not in ignore])

            key_map = {normalize(k): k for k in current_keys}
            
            final_weights = {}
            for k_raw, v_raw in raw_weights.items():
                k_norm = normalize(k_raw)
                if k_norm in key_map:
                    target_key = key_map[k_norm]
                    final_weights[target_key] = v_raw
            
            if final_weights:
                self.model.load_state_dict(final_weights, strict=False)
                print(f"✅ Loaded {len(final_weights)} layers.")
            else:
                print("⚠️  No matching layers found.")

            for name, param in self.model.named_parameters():
                if any(t in name for t in ["lora", "modules_to_save", "head", "projector"]): 
                    param.requires_grad = True

        except Exception as e:
            print(f"[ERROR] Weight loading failed: {e}")
            raise e