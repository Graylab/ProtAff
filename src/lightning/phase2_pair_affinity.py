import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional import spearman_corrcoef
from src.models import build_model 


def get_esm_lora_target_modules(esm_model, target_modules: list, n_last_layers: int = None):
    """
    Build LoRA target module list.
    
    Args:
        esm_model: The ESM model to get num_layers from
        target_modules: Base module names like ["query", "key", "value"]
        n_last_layers: If specified, only target last N layers. None = all layers
    
    Returns:
        List of target module patterns/names for LoRA
    """
    if n_last_layers is None:
        return target_modules
    
    num_layers = esm_model.config.num_hidden_layers
    full_target_modules = []
    
    for i in range(num_layers - n_last_layers, num_layers):
        for module in target_modules:
            full_target_modules.append(f"esm.encoder.layer.{i}.attention.self.{module}")
    
    return full_target_modules


class PairAffinityModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Model
        self.base_model = build_model(cfg)
        
        # 2. PEFT Setup
        modules_to_save = ["head_score"]
        arch = cfg.model.get("arch", "concat")
        
        if arch == "interaction_map":
            modules_to_save.extend(["norm_binder", "norm_target", "cross_attn", "norm_final", "projector"])
        elif arch == "cross_attn":
            modules_to_save.extend(["norm_binder", "norm_target", "cross_attn", "norm_final", "projector"])
        else:
            modules_to_save.extend(["projector", "norm_input"])
        
        if cfg.model.lora.get("modules_to_save"):
            extra = OmegaConf.to_container(cfg.model.lora.modules_to_save)
            for m in extra:
                if m not in modules_to_save:
                    modules_to_save.append(m)

        # Build target modules (with n_last_layers support)
        base_target_modules = OmegaConf.to_container(cfg.model.lora.target_modules)
        n_last_layers = cfg.model.lora.get("n_last_layers", None)
        
        target_modules = get_esm_lora_target_modules(
            self.base_model.esm, 
            base_target_modules, 
            n_last_layers
        )
        
        if n_last_layers is not None:
            print(f"[LoRA] Targeting last {n_last_layers} layers")
            print(f"[LoRA] Target modules: {target_modules[:3]}... ({len(target_modules)} total)")
        else:
            print(f"[LoRA] Targeting all layers with modules: {target_modules}")

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
        
        # 3. Loss
        self.margin = getattr(cfg.training, "margin", 0.1)
        self.rank_loss = nn.MarginRankingLoss(margin=self.margin)
        
        # 4. Validation storage for per-target Spearman
        self.val_scores = []
        self.val_target_ids = []
        self.val_binder_ids = []
        self.val_log_affs = []
        
        # 5. Phase transfer
        ckpt_path = cfg.get("pretrained_ckpt_path", None) 
        if ckpt_path:
            self._load_phase1_weights(ckpt_path)

    def forward(self, batch_subset, prefix="better"):
        arch = self.cfg.model.get("arch", "concat")
        is_test_batch = 'input_ids' in batch_subset or 'binder_ids' in batch_subset
        
        if is_test_batch:
            if arch in ["cross_attn", "interaction_map"]:
                return self.model(
                    binder_ids=batch_subset['binder_ids'],
                    binder_mask=batch_subset['binder_mask'],
                    target_ids=batch_subset['target_ids'],
                    target_mask=batch_subset['target_mask']
                )
            else:
                return self.model(
                    input_ids=batch_subset['input_ids'], 
                    attention_mask=batch_subset['attention_mask']
                )
        else:
            if arch in ["cross_attn", "interaction_map"]:
                return self.model(
                    binder_ids=batch_subset[f'{prefix}_binder_ids'],
                    binder_mask=batch_subset[f'{prefix}_binder_mask'],
                    target_ids=batch_subset[f'{prefix}_target_ids'],
                    target_mask=batch_subset[f'{prefix}_target_mask']
                )
            else:
                return self.model(
                    input_ids=batch_subset[f'{prefix}_input_ids'], 
                    attention_mask=batch_subset[f'{prefix}_mask']
                )

    def training_step(self, batch, batch_idx):
        scores_better = self.forward(batch, prefix="better")
        scores_worse = self.forward(batch, prefix="worse")
        
        target_rank = torch.full_like(scores_better, -1.0)
        loss = self.rank_loss(scores_better, scores_worse, target_rank)
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        scores_better = self.forward(batch, prefix="better")
        scores_worse = self.forward(batch, prefix="worse")
        
        target_rank = torch.full_like(scores_better, -1.0)
        loss = self.rank_loss(scores_better, scores_worse, target_rank)
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store for per-target Spearman
        self.val_scores.extend(scores_better.detach().cpu().squeeze().tolist())
        self.val_scores.extend(scores_worse.detach().cpu().squeeze().tolist())
        self.val_target_ids.extend(batch['better_tid'])
        self.val_target_ids.extend(batch['worse_tid'])
        self.val_binder_ids.extend(batch['better_bid'])
        self.val_binder_ids.extend(batch['worse_bid'])
        self.val_log_affs.extend(batch['better_log_aff'].cpu().tolist())
        self.val_log_affs.extend(batch['worse_log_aff'].cpu().tolist())
        
        return loss

    def on_validation_epoch_end(self):
        if not self.val_scores:
            return
        
        df = pd.DataFrame({
            'score': self.val_scores,
            'target_id': self.val_target_ids,
            'binder_id': self.val_binder_ids,
            'log_aff': self.val_log_affs
        })
        
        df = df.drop_duplicates(subset=['target_id', 'binder_id'], keep='first')
        
        target_spearmans = []
        min_samples = 5
        
        for tid, group in df.groupby('target_id'):
            if len(group) < min_samples:
                continue
            
            scores = torch.tensor(group['score'].values)
            affs = torch.tensor(group['log_aff'].values)
            
            sp = spearman_corrcoef(scores, affs)
            if not torch.isnan(sp):
                target_spearmans.append(sp.item())
        
        if target_spearmans:
            avg_spearman = torch.tensor(np.mean(target_spearmans), device=self.device)
            self.log('val_per_target_spearman', avg_spearman, prog_bar=True, sync_dist=True)
            
            if self.trainer.is_global_zero:
                print(f"\n[Val] Per-target Spearman: {avg_spearman:.4f} (from {len(target_spearmans)} targets)")
        
        self.val_scores.clear()
        self.val_target_ids.clear()
        self.val_binder_ids.clear()
        self.val_log_affs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.cfg.training.learning_rate, 
            weight_decay=self.cfg.training.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(total_steps * getattr(self.cfg.training, "warmup_ratio", 0.1))

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

    def _load_phase1_weights(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Path not found: {ckpt_path}. Training from scratch.")
            return

        print(f"\n[INFO] Loading Phase 1 (DMS) weights from {ckpt_path}")
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
                print(f"✅ Successfully transferred {len(final_weights)} DMS-trained layers.")
            
            for name, param in self.model.named_parameters():
                if any(t in name for t in ["lora", "modules_to_save", "head", "norm_binder", "norm_target"]): 
                    param.requires_grad = True

        except Exception as e:
            print(f"[ERROR] Weight loading failed: {e}")
            raise e