import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional import spearman_corrcoef

from src.models import build_model


class PairDMSModule(pl.LightningModule):
    """
    DMS Pre-training Module with Pairwise Ranking Loss.
    Uses cross-attention architecture: Mutant (binder) attends to Wildtype (target).
    
    Weight Transfer: All layers are named identically to PairAffinityModule,
    enabling direct transfer via `pretrained_ckpt_path` in Phase 2.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Model Initialization (SAME as PairAffinityModule)
        self.base_model = build_model(cfg)
        
        # 2. Setup PEFT (LoRA) - SAME structure as PairAffinityModule
        modules_to_save = ["head_score"]
        arch = cfg.model.get("arch", "cross_attn")  # Default to cross_attn for DMS
        
        if arch == "interaction_map":
            modules_to_save.extend(["norm_binder", "norm_target", "cross_attn", "norm_final", "projector"])
        elif arch == "cross_attn":
            modules_to_save.extend(["norm_binder", "norm_target", "cross_attn", "norm_final", "projector"])
        else:  # concat
            modules_to_save.extend(["projector", "norm_input"])
        
        if cfg.model.lora.get("modules_to_save"):
            extra = OmegaConf.to_container(cfg.model.lora.modules_to_save)
            for m in extra:
                if m not in modules_to_save:
                    modules_to_save.append(m)

        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            target_modules=OmegaConf.to_container(cfg.model.lora.target_modules),
            modules_to_save=modules_to_save,
            lora_dropout=cfg.model.lora.dropout,
            bias="none"
        )

        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()
        
        # 3. Ranking Loss - SAME as PairAffinityModule
        # Lower score = Better (stronger binding / higher fitness)
        self.margin = getattr(cfg.training, "margin", 0.1)
        self.rank_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, batch_subset, prefix="better"):
        """
        Architecture-aware forward pass - SAME signature as PairAffinityModule.
        
        For DMS:
        - binder = mutant sequence
        - target = wildtype sequence
        """
        arch = self.cfg.model.get("arch", "cross_attn")
        
        if arch in ["cross_attn", "interaction_map"]:
            return self.model(
                binder_ids=batch_subset[f'{prefix}_binder_ids'],
                binder_mask=batch_subset[f'{prefix}_binder_mask'],
                target_ids=batch_subset[f'{prefix}_target_ids'],
                target_mask=batch_subset[f'{prefix}_target_mask']
            )
        else:  # concat
            return self.model(
                input_ids=batch_subset[f'{prefix}_input_ids'],
                attention_mask=batch_subset[f'{prefix}_mask']
            )

    def training_step(self, batch, batch_idx):
        # Forward pass - SAME as PairAffinityModule
        scores_better = self.forward(batch, prefix="better")
        scores_worse = self.forward(batch, prefix="worse")
        
        # Ranking loss: Better should have LOWER score
        # target_rank = -1 ensures scores_better < scores_worse
        target_rank = torch.full_like(scores_better, -1.0)
        loss = self.rank_loss(scores_better, scores_worse, target_rank)
        
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        scores_better = self.forward(batch, prefix="better")
        scores_worse = self.forward(batch, prefix="worse")
        
        target_rank = torch.full_like(scores_better, -1.0)
        loss = self.rank_loss(scores_better, scores_worse, target_rank)
        accuracy = (scores_better < scores_worse).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"loss": loss, "scores_better": scores_better, "scores_worse": scores_worse}

    def configure_optimizers(self):
        # SAME structure as PairAffinityModule
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