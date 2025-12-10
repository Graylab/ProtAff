import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from src.models.model_base import ESMCrossAttentionClassifier

class ProteinAffinityModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Container
        base_model = ESMCrossAttentionClassifier(cfg.model.name, cfg=cfg)
        
        # 2. Setup PEFT (Training Decoders + Encoder + Poolers)
        custom_modules = ["projector", "norm_input", "task_encoder", "decoder_b2t", "decoder_t2b", "norm_pooled_b", "norm_pooled_t", "head_score"]
        if getattr(cfg.model, "pooling", "mean") == "attention":
            custom_modules.extend(["pooler_b", "pooler_t"])
        
        target_modules = OmegaConf.to_container(cfg.model.lora.target_modules)
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

        self.model = get_peft_model(base_model, peft_config)
        self.loss_fn = torch.nn.MSELoss()

        # ----------------------------------------------------------------------
        # TRANSFER LEARNING LOADING
        # ----------------------------------------------------------------------

        # [VERIFICATION STEP 1] Capture Random Fingerprint
        # We grab the norm of the Projector weights. If loading works, this MUST change.
        with torch.no_grad():
            init_proj = self.model.base_model.model.projector.weight.norm().item()

        ckpt_path = cfg.get("pretrained_ckpt_path", None)
        
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\n[INFO] Loading Phase 1 weights from {ckpt_path}")
            try:
                raw_weights = load_peft_weights(ckpt_path, device="cpu")
                
                # --- UNIVERSAL KEY MATCHER ---
                def normalize_key(k):
                    parts = k.split('.')
                    clean = [p for p in parts if p not in ['base_model', 'model', 'module', 'original_module', 'modules_to_save', 'default']]
                    return '.'.join(clean)

                # Map Current Model Keys
                model_map = {normalize_key(k): k for k in self.model.state_dict().keys()}
                
                # Match Checkpoint Keys
                final_weights = {}
                for k, v in raw_weights.items():
                    if "single_head" in k: continue 
                    clean_k = normalize_key(k)
                    if clean_k in model_map:
                        final_weights[model_map[clean_k]] = v

                # Load
                missing, unexpected = self.model.load_state_dict(final_weights, strict=False)
                
                # Symmetrize
                print("[INFO] Copying Binder Weights to Target...")
                with torch.no_grad():
                    bm = self.model.base_model.model
                    if hasattr(bm, "pooler_t") and hasattr(bm, "pooler_b"):
                        bm.pooler_t.load_state_dict(bm.pooler_b.state_dict())
                    bm.norm_pooled_t.load_state_dict(bm.norm_pooled_b.state_dict())

                print(f"✅ SUCCESS: Phase 1 weights loaded into Encoder.")

                # [VERIFICATION STEP 2] Compare Fingerprints
                with torch.no_grad():
                    new_proj = self.model.base_model.model.projector.weight.norm().item()
                
                print(f"[DEBUG] Fingerprint Check (Projector Norm): {init_proj:.4f} -> {new_proj:.4f}")
                
                if init_proj != new_proj:
                    print("✅ SUCCESS: Weights updated successfully.")
                else:
                    print("❌ ERROR: Weights did NOT change. Loading failed!")
                
                # Unfreeze
                for name, param in self.model.named_parameters():
                    if any(t in name for t in ["lora", "modules_to_save"]): 
                        param.requires_grad = True

            except Exception as e:
                print(f"[ERROR] Loading failed: {e}")
        else:
            print("[INFO] Training from Scratch.")

    # --- PHASE 2 FORWARD ---
    def forward(self, batch):
        bm = self.model.base_model.model
        
        # 1. ESM
        b_raw = bm.esm(batch['binder_ids'], attention_mask=batch['binder_mask']).last_hidden_state
        t_raw = bm.esm(batch['target_ids'], attention_mask=batch['target_mask']).last_hidden_state
        
        # 2. Project
        b_vec = bm.norm_input(bm.projector(b_raw))
        t_vec = bm.norm_input(bm.projector(t_raw))
        
        b_mask = ~batch['binder_mask'].bool()
        t_mask = ~batch['target_mask'].bool()
        
        # 3. Encoder (PRE-TRAINED)
        b_enc = bm.task_encoder(b_vec, src_key_padding_mask=b_mask)
        t_enc = bm.task_encoder(t_vec, src_key_padding_mask=t_mask)
        
        # 4. Decoder (NEW / FINETUNED)
        # The decoder takes the ENCODED features as input/memory
        dec_b = bm.decoder_b2t(b_enc, memory=t_enc, tgt_key_padding_mask=b_mask, memory_key_padding_mask=t_mask)
        dec_t = bm.decoder_t2b(t_enc, memory=b_enc, tgt_key_padding_mask=t_mask, memory_key_padding_mask=b_mask)
        
        # 5. Pool (PRE-TRAINED)
        # Note: We now pool the DECODER output.
        # This works because the pooler expects (B, L, D) and learned "importance"
        # The Decoder preserves sequence length and dim, so the transfer is valid.
        b_p_mask = bm.get_pooling_mask(batch['binder_ids'], batch['binder_mask'])
        t_p_mask = bm.get_pooling_mask(batch['target_ids'], batch['target_mask'])
        
        if bm.pooling_type == "attention":
            pooled_b = bm.pooler_b(dec_b, b_p_mask)
            pooled_t = bm.pooler_t(dec_t, t_p_mask)
        else:
            pooled_b = (dec_b * b_p_mask).sum(dim=1) / torch.clamp(b_p_mask.sum(dim=1), min=1e-9)
            pooled_t = (dec_t * t_p_mask).sum(dim=1) / torch.clamp(t_p_mask.sum(dim=1), min=1e-9)
            
        pooled_b = bm.norm_pooled_b(pooled_b)
        pooled_t = bm.norm_pooled_t(pooled_t)
        
        # 6. Head
        return bm.head_score(torch.cat([pooled_b, pooled_t], dim=-1)).squeeze(-1)

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self(batch), batch['labels'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_fn(self(batch), batch['labels'])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 1. Optimizer (AdamW)
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.training.learning_rate, 
            weight_decay=self.cfg.training.weight_decay
        )

        # 2. Calculate Total Steps
        # (Lightning helper to get total number of batches across all epochs)
        total_steps = self.trainer.estimated_stepping_batches
        
        # 3. Warmup Calculation
        # The standard convention (RoBERTa/BERT/LoRA) is ~6% to 10% of total steps.
        # If your config doesn't specify it, default to 10% (0.1).
        warmup_ratio = getattr(self.cfg.training, "warmup_ratio", 0.1)
        num_warmup_steps = int(total_steps * warmup_ratio)

        # 4. Scheduler: Linear Warmup -> Linear Decay
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # IMPORTANT: Update every batch, not every epoch
                "frequency": 1
            }
        }
