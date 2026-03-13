import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.models import build_model


def get_esm_lora_target_modules(esm_model, target_modules: list, n_last_layers: int = None):
    if n_last_layers is None:
        return target_modules
    num_layers = esm_model.config.num_hidden_layers
    return [
        f"esm.encoder.layer.{i}.attention.self.{m}"
        for i in range(num_layers - n_last_layers, num_layers)
        for m in target_modules
    ]


class BaseModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        # 1. Base model
        self.base_model = build_model(cfg)

        # 2. PEFT setup
        self.model = self._setup_peft(cfg)


    def _setup_peft(self, cfg):
        use_lora = cfg.model.get("use_lora", True)

        if not use_lora:
            # Frozen ESM2 baseline: freeze backbone, train only custom layers
            print("[Baseline] LoRA disabled — freezing ESM2 backbone")
            for param in self.base_model.esm.parameters():
                param.requires_grad = False

            trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.base_model.parameters())
            print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")
            return self.base_model

        n_cross = getattr(cfg.model, "n_cross_layers", 1)

        modules_to_save = ["pool", "input_norm", "input_proj", "head_affinity"]
        modules_to_save.extend([f"cross_layers.{i}" for i in range(n_cross)])

        if cfg.model.lora.get("modules_to_save"):
            for m in OmegaConf.to_container(cfg.model.lora.modules_to_save):
                if m not in modules_to_save:
                    modules_to_save.append(m)

        base_target_modules = OmegaConf.to_container(cfg.model.lora.target_modules)
        n_last_layers = cfg.model.lora.get("n_last_layers", None)
        target_modules = get_esm_lora_target_modules(
            self.base_model.esm, base_target_modules, n_last_layers
        )

        if n_last_layers is not None:
            print(f"[LoRA] Targeting last {n_last_layers} layers ({len(target_modules)} modules)")
        else:
            print(f"[LoRA] Targeting all layers with: {target_modules}")

        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
        )

        model = get_peft_model(self.base_model, peft_config)
        model.print_trainable_parameters()
        return model

    def _forward_single(self, batch, prefix=None):
        if prefix:
            keys = lambda k: f"{prefix}_{k}"
        else:
            keys = lambda k: k

        return self.model(
            binder_ids=batch[keys("binder_ids")],
            binder_mask=batch[keys("binder_mask")],
            target_ids=batch[keys("target_ids")],
            target_mask=batch[keys("target_mask")],
        )

    def forward(self, batch, prefix=None):
        return self._forward_single(batch, prefix)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = int(total_steps * getattr(self.cfg.training, "warmup_ratio", 0.1))

        schedule_type = getattr(self.cfg.training, "scheduler", "cosine")
        if schedule_type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

