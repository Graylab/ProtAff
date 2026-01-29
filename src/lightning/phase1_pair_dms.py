import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup

from src.models import build_model


def get_esm_lora_target_modules(esm_model, target_modules: list, n_last_layers: int = None):
    if n_last_layers is None:
        return target_modules
    
    num_layers = esm_model.config.num_hidden_layers
    full_target_modules = []
    
    for i in range(num_layers - n_last_layers, num_layers):
        for module in target_modules:
            full_target_modules.append(f"esm.encoder.layer.{i}.attention.self.{module}")
    
    return full_target_modules


class PairDMSModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Model
        self.base_model = build_model(cfg)
        
        # 2. PEFT Setup
        modules_to_save = ["head_score"]
        arch = cfg.model.get("arch", "cross_attn")
        
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

        # Build target modules (with n_last_layers support)  # ADDED
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
            target_modules=target_modules,  # CHANGED
            modules_to_save=modules_to_save,
            lora_dropout=cfg.model.lora.dropout,
            bias="none"
        )

        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()
        
        # 3. Loss
        self.margin = getattr(cfg.training, "margin", 0.1)
        self.loss_type = getattr(cfg.training, "loss_type", "margin")  # ADDED

    def _compute_loss(self, scores_better, scores_worse, delta=None):  # ADDED
        if self.loss_type == "margin_weighted" and delta is not None:
            target_rank = torch.full_like(scores_better, -1.0)
            loss = nn.functional.margin_ranking_loss(
                scores_better, scores_worse, target_rank, 
                margin=self.margin, reduction='none'
            )
            weights = torch.clamp(delta, min=0.1, max=5.0)
            return (loss * weights.unsqueeze(-1)).mean()
        else:
            target_rank = torch.full_like(scores_better, -1.0)
            return nn.functional.margin_ranking_loss(
                scores_better, scores_worse, target_rank, 
                margin=self.margin
            )

    def forward(self, batch_subset, prefix="better"):
        arch = self.cfg.model.get("arch", "cross_attn")
        
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
        
        delta = batch.get('delta', None)
        loss = self._compute_loss(scores_better, scores_worse, delta)
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        scores_better = self.forward(batch, prefix="better")
        scores_worse = self.forward(batch, prefix="worse")
        
        delta = batch.get('delta', None)
        loss = self._compute_loss(scores_better, scores_worse, delta)
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }