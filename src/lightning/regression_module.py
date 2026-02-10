import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef

from src.lightning.base_module import BaseModule, FocalLoss


class RegressionModule(BaseModule):
    """
    Unified regression module for affinity, DMS, and PPI tasks.
    Supports loss types: mse, smooth_l1, huber, bce, focal.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

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
            gamma = getattr(cfg.training, "focal_gamma", 2.0)
            alpha = getattr(cfg.training, "focal_alpha", 1.0)
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.loss_fn = nn.MSELoss()

        print(f"[RegressionModule] Loss: {self.loss_type}")
        if self.loss_type in ["bce", "focal"]:
            print(f"[RegressionModule] Threshold: {self.bce_threshold}")

    def _get_labels(self, batch):
        """Get labels from batch, converting for BCE/Focal if needed."""
        labels = batch["reg_labels"]

        if self.loss_type in ["bce", "focal"]:
            labels = labels.reshape(-1)
            binary_labels = (labels > (1.0 - self.bce_threshold)).float()
            return binary_labels

        return labels

    def training_step(self, batch, batch_idx):
        preds = self._forward_single(batch)
        labels = self._get_labels(batch)

        # Squeeze for BCE/Focal which expects matching shapes
        if self.loss_type in ["bce", "focal"]:
            preds = preds.squeeze(-1)

        loss = self.loss_fn(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.loss_type in ["bce", "focal"]:
            acc = ((preds > 0) == (labels > 0.5)).float().mean()
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self._forward_single(batch)
        labels = self._get_labels(batch)

        if self.loss_type in ["bce", "focal"]:
            preds = preds.squeeze(-1)

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
            spearman = spearman_corrcoef(preds.reshape(-1), labels.reshape(-1))
            self.log("val_spearman", spearman, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
