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
        
        # 3. Loss Function (Ranking)
        # MarginRankingLoss: max(0, -y * (x1 - x2) + margin)
        # We default margin to 0.1 if not specified
        self.margin = getattr(cfg.training, "margin", 0.1)
        self.rank_loss = nn.MarginRankingLoss(margin=self.margin)

        # Used to anchor scores: "Binders" -> 1.0, "Non-Binders" -> 0.0
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.cls_weight = getattr(cfg.training, "cls_weight", 0.5)
        
        # 4. Transfer Learning
        ckpt_path = cfg.get("pretrained_ckpt_path", None) 
        if ckpt_path:
            self._load_phase1_weights(ckpt_path)

    def forward(self, input_ids, attention_mask):
        """
        Standard forward pass for a single batch of sequences.
        Useful for inference to get a raw score.
        """
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

    def training_step(self, batch, batch_idx):
        # 1. Forward Pass A (Better Samples)
        scores_better = self(
            input_ids=batch['better_input_ids'], 
            attention_mask=batch['better_mask']
        )
        
        scores_worse = self(
            input_ids=batch['worse_input_ids'], 
            attention_mask=batch['worse_mask']
        )
        
        # A. Ranking Loss (Existing)
        target_rank = torch.ones_like(scores_better)
        loss_rank = self.rank_loss(scores_better, scores_worse, target_rank)

        # B. Classification Loss (New)
        loss_cls_better = self.cls_loss(scores_better.view(-1), batch['better_labels'])
        loss_cls_worse = self.cls_loss(scores_worse.view(-1), batch['worse_labels'])
        loss_cls = (loss_cls_better + loss_cls_worse) / 2.0
        
        # C. Total Loss
        loss = loss_rank + (self.cls_weight * loss_cls)

        # --- LOGGING UPDATES ---
        accuracy = (scores_better > scores_worse).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_rank', loss_rank, on_epoch=True) # Monitor Rank split
        self.log('train_loss_cls', loss_cls, on_epoch=True)   # Monitor Cls split
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward Pass
        scores_better = self(
            input_ids=batch['better_input_ids'], 
            attention_mask=batch['better_mask']
        )
        scores_worse = self(
            input_ids=batch['worse_input_ids'], 
            attention_mask=batch['worse_mask']
        )
        
        # A. Ranking Loss
        target_rank = torch.ones_like(scores_better)
        loss_rank = self.rank_loss(scores_better, scores_worse, target_rank)
        
        # B. Classification Loss
        loss_cls_better = self.cls_loss(scores_better.view(-1), batch['better_labels'])
        loss_cls_worse = self.cls_loss(scores_worse.view(-1), batch['worse_labels'])
        loss_cls = (loss_cls_better + loss_cls_worse) / 2.0
        
        # C. Total Loss
        loss = loss_rank + (self.cls_weight * loss_cls)
        
        # --- LOGGING UPDATES ---
        accuracy = (scores_better > scores_worse).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_loss_rank', loss_rank, on_epoch=True, sync_dist=True) # Monitor Rank split
        self.log('val_loss_cls', loss_cls, on_epoch=True, sync_dist=True)   # Monitor Cls split
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