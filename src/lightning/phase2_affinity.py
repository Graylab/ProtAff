import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional import spearman_corrcoef

# Architecture factory
from src.models import build_model 

class AffinityModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Model Initialization
        self.base_model = build_model(cfg)
        
        # 2. Setup PEFT (LoRA)
        # We ensure the interaction-specific norms and head are trainable
        modules_to_save = ["head_score"]
        arch = cfg.model.get("arch", "concat")
        
        if arch == "interaction_map":
            # Saves normalization layers for the independent ESM encodings
            modules_to_save.extend(["norm_binder", "norm_target"])
        elif arch == "cross_attn":
            modules_to_save.extend(["norm_binder", "norm_target", "cross_attn", "norm_final", "projector"])
        else: # concat
            modules_to_save.extend(["projector", "norm_input"])
        
        if cfg.model.lora.get("modules_to_save", None):
             extra = OmegaConf.to_container(cfg.model.lora.modules_to_save)
             for m in extra:
                 if m not in modules_to_save:
                     modules_to_save.append(m)

        # 3. LoRA Integration
        # Maintains original variable mapping: cfg.model.lora.alpha
        print(f"[Model] Initializing LoRA (r={cfg.model.lora.r}) for {arch}")
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

        # 4. Loss Function
        self.mse_loss = nn.MSELoss() 
        
        # 5. Pretrained Weights Loading
        ckpt_path = cfg.get("pretrained_ckpt_path", None) 
        if ckpt_path:
            self._load_phase1_weights(ckpt_path)

    def forward(self, batch):
        """
        Architecture-aware forward pass routing.
        """
        arch = self.cfg.model.get("arch", "concat")
        
        if arch in ["cross_attn", "interaction_map"]:
            # Direct mapping to ESMCrossAttnModel.forward signature
            return self.model(
                binder_ids=batch['binder_ids'],
                binder_mask=batch['binder_mask'],
                target_ids=batch['target_ids'],
                target_mask=batch['target_mask']
            )
        else:
            # Standard ESMConcatModel.forward signature
            return self.model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )

    def training_step(self, batch, batch_idx):
        pred_reg = self.forward(batch)
        loss = self.mse_loss(pred_reg, batch['reg_labels'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_reg = self.forward(batch)
        reg_labels = batch['reg_labels']
        loss = self.mse_loss(pred_reg, reg_labels)
        
        # Scientific metric for Novel Target ranking
        spearman = spearman_corrcoef(pred_reg.reshape(-1), reg_labels.reshape(-1))
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_spearman', spearman, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.cfg.training.learning_rate, 
            weight_decay=self.cfg.training.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(total_steps * getattr(self.cfg.training, "warmup_ratio", 0.1))

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
            print(f"[WARN] Path not found: {ckpt_path}")
            return

        print(f"\n[INFO] Loading LoRA weights from Phase 1: {ckpt_path}")
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
                    final_weights[key_map[k_norm]] = v_raw
            
            if final_weights:
                self.model.load_state_dict(final_weights, strict=False)
                print(f"✅ Loaded {len(final_weights)} matching layers.")
            
            # Re-verify gradient requirements for fine-tuning
            for name, param in self.model.named_parameters():
                if any(t in name for t in ["lora", "modules_to_save", "head", "projector", "cross_attn"]): 
                    param.requires_grad = True

        except Exception as e:
            print(f"[ERROR] Weight loading failed: {e}")
            raise e