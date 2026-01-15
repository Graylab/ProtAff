import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional import spearman_corrcoef

# Assuming this function returns an ESM2 model with a scalar regression head
from src.models import build_model 

class AffinityModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Container 
        self.base_model = build_model(cfg)
        
        # Define trainable head layers
        modules_to_save = ["projector", "norm_input", "head_score"]
        
        # Append extra modules from config if they exist
        if cfg.model.lora.get("modules_to_save", None):
             extra = OmegaConf.to_container(cfg.model.lora.modules_to_save)
             for m in extra:
                 if m not in modules_to_save:
                     modules_to_save.append(m)

        # 2. Setup Model Architecture
        if cfg.model.lora.r > 0:
            # LoRA Mode
            print(f"[Model] Initializing LoRA (r={cfg.model.lora.r})...")
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
            self.model.print_trainable_parameters()

        else:
            # Linear Probe Mode
            print("[Model] Linear Probe Mode: LoRA Disabled (r=0).")
            self.model = self.base_model
            
            # Freeze entire model first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Manually unfreeze head layers
            trainable_params = 0
            all_params = 0
            
            print("[Model] Manually unfreezing head layers:")
            for name, param in self.model.named_parameters():
                all_params += param.numel()
                if any(layer in name for layer in modules_to_save):
                    param.requires_grad = True
                    trainable_params += param.numel()
                    print(f"  -> Unfrozen: {name}")
            
            print(f"trainable params: {trainable_params:,} || all params: {all_params:,}")

        # 3. Loss Function
        self.mse_loss = nn.MSELoss() 
        
        # 4. Transfer Learning
        ckpt_path = cfg.get("pretrained_ckpt_path", None) 
        if ckpt_path:
            self._load_phase1_weights(ckpt_path)

    def forward(self, batch):
        return self.model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask']
        )

    def training_step(self, batch, batch_idx):
        pred_reg = self(batch)
        reg_labels = batch['reg_labels'] 
        
        loss = self.mse_loss(pred_reg, reg_labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_reg = self(batch)
        reg_labels = batch['reg_labels']
        
        loss = self.mse_loss(pred_reg, reg_labels)
        
        # Scientific Validation Metric: Spearman Correlation
        # This checks if the ranking of predicted affinities matches reality
        # even if the absolute values are off.
        spearman = spearman_corrcoef(pred_reg.squeeze(), reg_labels.squeeze())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_spearman', spearman, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Filter to only optimize parameters that require gradients (LoRA + Head)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
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

        print(f"\n[INFO] Loading weights from {ckpt_path}")
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

            # Ensure gradients are enabled for LoRA/Head after loading
            for name, param in self.model.named_parameters():
                if any(t in name for t in ["lora", "modules_to_save", "head", "projector"]): 
                    param.requires_grad = True

        except Exception as e:
            print(f"[ERROR] Weight loading failed: {e}")
            raise e