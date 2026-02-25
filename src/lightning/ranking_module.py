import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.lightning.base_module import BaseModule


class RankingModule(BaseModule):
    """
    Unified ranking module for pair_affinity and pair_ppi tasks.
    Supports loss types: margin, soft_margin, bce, margin_weighted, contrastive.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.margin = getattr(cfg.training, "margin", 0.1)
        self.loss_type = getattr(cfg.training, "loss_type", "margin")

        if self.loss_type == "margin":
            self.rank_loss = nn.MarginRankingLoss(margin=self.margin)
        elif self.loss_type == "soft_margin":
            self.rank_loss = nn.SoftMarginLoss()
        elif self.loss_type == "bce":
            self.rank_loss = nn.BCEWithLogitsLoss()

        print(f"[RankingModule] Loss: {self.loss_type}, Margin: {self.margin}")

    def _compute_loss(self, scores_better, scores_worse, delta=None):
        # Convention: scores_better > scores_worse (higher predicted_affinity = better binder)
        if self.loss_type == "margin":
            target = torch.full_like(scores_better, 1.0)
            return self.rank_loss(scores_better, scores_worse, target)

        elif self.loss_type in ["soft_margin", "bce"]:
            diff = scores_better - scores_worse
            target = torch.ones_like(diff)
            return self.rank_loss(diff, target)

        elif self.loss_type == "margin_weighted":
            target = torch.full_like(scores_better, 1.0)
            loss = F.margin_ranking_loss(
                scores_better, scores_worse, target,
                margin=self.margin, reduction="none"
            )
            if delta is not None:
                loss = loss * torch.clamp(delta, min=0.1, max=5.0).unsqueeze(-1)
            return loss.mean()

        elif self.loss_type == "contrastive":
            diff = scores_better - scores_worse
            temp = getattr(self.cfg.training, "temperature", 0.1)
            return -torch.log(torch.sigmoid(diff / temp) + 1e-8).mean()

        else:
            # Fallback: standard margin
            target = torch.full_like(scores_better, 1.0)
            return self.rank_loss(scores_better, scores_worse, target)

    def forward(self, batch, prefix="better"):
        """Forward pass handling both prefixed training batches and unprefixed test batches."""
        is_test_batch = "input_ids" in batch or "binder_ids" in batch

        if is_test_batch:
            return self._forward_single(batch, prefix=None)
        else:
            return self._forward_single(batch, prefix=prefix)

    def training_step(self, batch, batch_idx):
        scores_better = self._forward_single(batch, prefix="better")
        scores_worse = self._forward_single(batch, prefix="worse")

        delta = batch.get("delta", None)
        loss = self._compute_loss(scores_better, scores_worse, delta)
        accuracy = (scores_better > scores_worse).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        scores_better = self._forward_single(batch, prefix="better")
        scores_worse = self._forward_single(batch, prefix="worse")

        delta = batch.get("delta", None)
        loss = self._compute_loss(scores_better, scores_worse, delta)
        accuracy = (scores_better > scores_worse).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
