import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef

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


# ----------------------------------------------------------------------
# Focal Loss Implementation
# ----------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * (1 - p_t)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class PPIModule(pl.LightningModule):
    """Lightning module for PPI pretraining (regression or binary classification)."""
    
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
        self.loss_type = getattr(cfg.training, "loss_type", "mse")
        self.bce_threshold = getattr(cfg.data, "confidence_threshold", 0.5)
        
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        elif self.loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        elif self.loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.loss_type == "focal":
            # Added Focal Loss
            gamma = getattr(cfg.training, "focal_gamma", 2.0)
            alpha = getattr(cfg.training, "focal_alpha", 1.0)
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.loss_fn = nn.MSELoss()
        
        print(f"[PPIModule] Loss: {self.loss_type}")
        if self.loss_type in ["bce", "focal"]:
            print(f"[PPIModule] Threshold: {self.bce_threshold}")

        # 4. Phase transfer
        ckpt_path = cfg.get("pretrained_ckpt_path", None) 
        if ckpt_path:
            self._load_phase1_weights(ckpt_path)

    def forward(self, batch):
        arch = self.cfg.model.get("arch", "concat")
        
        if arch in ["cross_attn", "interaction_map"]:
            return self.model(
                binder_ids=batch["binder_ids"],
                binder_mask=batch["binder_mask"],
                target_ids=batch["target_ids"],
                target_mask=batch["target_mask"]
            )
        else:
            return self.model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"]
            )

    def _get_labels(self, batch):
        """Convert labels for BCE or Focal if needed."""
        labels = batch["labels"]
        
        if self.loss_type in ["bce", "focal"]:
            # labels are inverted confidence (1 - conf) if invert_label=True
            # For BCE/Focal: binder (high conf, low label) = 0, non-binder (low conf, high label) = 1
            # threshold on inverted: label > (1 - threshold) means non-binder
            binary_labels = (labels > (1.0 - self.bce_threshold)).float()
            return binary_labels
        
        return labels

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch).squeeze(-1)
        labels = self._get_labels(batch)
        
        loss = self.loss_fn(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if self.loss_type in ["bce", "focal"]:
            acc = ((preds > 0) == (labels > 0.5)).float().mean()
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch).squeeze(-1)
        labels = self._get_labels(batch)
        
        loss = self.loss_fn(preds, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if self.loss_type in ["bce", "focal"]:
            acc = ((preds > 0) == (labels > 0.5)).float().mean()
            
            pred_binder = (preds <= 0)
            true_binder = (labels < 0.5)
            
            tp = (pred_binder & true_binder).sum().float()
            fp = (pred_binder & ~true_binder).sum().float()
            fn = (~pred_binder & true_binder).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_precision", precision, on_epoch=True, sync_dist=True)
            self.log("val_recall", recall, on_epoch=True, sync_dist=True)
            self.log("val_f1", f1, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            spearman = spearman_corrcoef(preds, labels)
            pearson = pearson_corrcoef(preds, labels)
            self.log("val_spearman", spearman, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_pearson", pearson, on_epoch=True, prog_bar=True, sync_dist=True)
        
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

        print(f"\n[INFO] Loading pretrained weights from {ckpt_path}")
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
                print(f"✅ Successfully transferred {len(final_weights)} pretrained layers.")
            
            for name, param in self.model.named_parameters():
                if any(t in name for t in ["lora", "modules_to_save", "head", "norm_binder", "norm_target"]): 
                    param.requires_grad = True

        except Exception as e:
            print(f"[ERROR] Weight loading failed: {e}")
            raise e