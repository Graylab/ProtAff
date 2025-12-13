import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from src.models import build_model

class AffinityModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Container (Uses Factory)
        self.base_model = build_model(cfg)
        
        # 2. Setup PEFT (LoRA)
        # Determine architecture to decide what custom modules to save
        arch = getattr(cfg.model, "arch", "cross_attn")
        
        if arch == "concat":
            # Concat Model: Only Projector and Head need saving
            custom_modules = ["projector", "norm_input", "head_score"]
        else:
            # Dual/Cross-Attn Model: Needs Encoders, Decoders, Poolers
            custom_modules = ["projector", "norm_input", "task_encoder", 
                              "decoder_b2t", "decoder_t2b", 
                              "norm_pooled_b", "norm_pooled_t", 
                              "head_score"]
            if getattr(cfg.model, "pooling", "mean") == "attention":
                custom_modules.extend(["pooler_b", "pooler_t"])
        
        target_modules = OmegaConf.to_container(cfg.model.lora.target_modules)
        
        # Ensure custom modules are in 'modules_to_save'
        modules_to_save = OmegaConf.to_container(cfg.model.lora.modules_to_save) if cfg.model.lora.modules_to_save else []
        for mod in custom_modules:
            if mod not in modules_to_save: modules_to_save.append(mod)

        peft_config = LoraConfig(
            r=cfg.model.lora.r, 
            lora_alpha=cfg.model.lora.alpha, 
            target_modules=target_modules,
            modules_to_save=modules_to_save, 
            lora_dropout=cfg.model.lora.dropout, 
            bias="none"
        )

        self.model = get_peft_model(self.base_model, peft_config)
        self.loss_fn = torch.nn.MSELoss()

        # ----------------------------------------------------------------------
        # 3. TRANSFER LEARNING (Load Phase 1)
        # ----------------------------------------------------------------------
        ckpt_path = cfg.get("pretrained_ckpt_path", None) 
        
        if ckpt_path:
            self._load_phase1_weights(ckpt_path)

    def _load_phase1_weights(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Path not found: {ckpt_path}. Training from scratch.")
            return

        print(f"\n[INFO] Loading Phase 1 weights from {ckpt_path}")
        
        try:
            # 1. Capture Fingerprint
            with torch.no_grad():
                # Access innermost model safely
                # Concat model has same structure: base_model.model.projector
                init_proj = self.model.base_model.model.projector.weight.norm().item()

            # 2. Load Raw Weights
            raw_weights = load_peft_weights(ckpt_path, device="cpu")
            
            # --- UNIVERSAL KEY MATCHER ---
            ignore_list = [
                'base_model', 'model', 'module', 'original_module', 
                'modules_to_save', 'default', 'base_layer'
            ]

            def normalize_key(k):
                parts = k.split('.')
                clean = [p for p in parts if p not in ignore_list]
                return '.'.join(clean)

            # A. Map Current Model Keys
            model_map = {normalize_key(k): k for k in self.model.state_dict().keys()}
            
            # B. Match Checkpoint Keys
            final_weights = {}
            for k, v in raw_weights.items():
                if "single_head" in k: continue 
                
                clean_k = normalize_key(k)
                if clean_k in model_map:
                    real_key = model_map[clean_k]
                    final_weights[real_key] = v

            # 3. Load
            if not final_weights:
                print("[WARN] No keys matched (Normal if architecture changed significantly).")
            else:
                self.model.load_state_dict(final_weights, strict=False)
            
            # 4. Symmetrize (Only for Dual/Siamese models)
            print("[INFO] Checking for Siamese weights to clone...")
            with torch.no_grad():
                bm = self.model.base_model.model
                # hasattr check prevents crash on Concat model
                if hasattr(bm, "pooler_t") and hasattr(bm, "pooler_b"):
                    bm.pooler_t.load_state_dict(bm.pooler_b.state_dict())
                    print("   -> Cloned pooler_b to pooler_t")
                if hasattr(bm, "norm_pooled_t") and hasattr(bm, "norm_pooled_b"):
                    bm.norm_pooled_t.load_state_dict(bm.norm_pooled_b.state_dict())
                    print("   -> Cloned norm_pooled_b to norm_pooled_t")

            # 5. Verify Fingerprint
            with torch.no_grad():
                new_proj = self.model.base_model.model.projector.weight.norm().item()
            
            print(f"[DEBUG] Fingerprint Check (Projector Norm): {init_proj:.4f} -> {new_proj:.4f}")
            
            if abs(init_proj - new_proj) > 1e-6:
                print("✅ SUCCESS: Weights loaded.")
            else:
                print("⚠️ WARN: Weights loaded but values didn't change.")

            # 6. Unfreeze
            for name, param in self.model.named_parameters():
                if any(t in name for t in ["lora", "modules_to_save"]): 
                    param.requires_grad = True

        except Exception as e:
            print(f"[ERROR] Loading failed: {e}")
            raise e

    def forward(self, batch):
        # Delegate to base_model (handles both Concat and Dual logic internally)
        preds = self.model.base_model(
            binder_ids=batch['binder_ids'], 
            binder_mask=batch['binder_mask'],
            target_ids=batch['target_ids'], 
            target_mask=batch['target_mask']
        )
        return preds.squeeze(-1)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.loss_fn(preds, batch['labels'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.loss_fn(preds, batch['labels'])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
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