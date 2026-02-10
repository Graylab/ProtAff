import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from peft.utils import load_peft_weights
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


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return (self.alpha * (1 - p_t) ** self.gamma * bce).mean()


class BaseModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        # 1. Base model
        self.base_model = build_model(cfg)

        # 2. PEFT setup
        self.model = self._setup_peft(cfg)

        # 3. Pretrained weights
        ckpt_path = cfg.get("pretrained_ckpt_path", None)
        if ckpt_path:
            self._load_pretrained_weights(ckpt_path)

    def _setup_peft(self, cfg):
        arch = cfg.model.get("arch", "concat")
        n_cross = getattr(cfg.model, "n_cross_layers", 1)

        modules_to_save = ["head_score", "pool"]
        if arch in ["bi_cross_attn", "cross_attn", "binding"]:
            modules_to_save.extend(["input_norm", "input_proj"])
            modules_to_save.extend([f"cross_layers.{i}" for i in range(n_cross)])
            if arch == "binding":
                modules_to_save.extend(["head_classify", "head_affinity"])
        else:
            modules_to_save.extend(["projector", "norm_input"])

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
        arch = self.cfg.model.get("arch", "concat")

        if prefix:
            keys = lambda k: f"{prefix}_{k}"
        else:
            keys = lambda k: k

        if arch in ["cross_attn", "bi_cross_attn", "binding"]:
            kwargs = dict(
                binder_ids=batch[keys("binder_ids")],
                binder_mask=batch[keys("binder_mask")],
                target_ids=batch[keys("target_ids")],
                target_mask=batch[keys("target_mask")],
            )
            if arch == "binding":
                kwargs["task"] = self.cfg.model.get("binding_head", "affinity")
            return self.model(**kwargs)
        else:
            return self.model(
                input_ids=batch[keys("input_ids")],
                attention_mask=batch.get(keys("attention_mask"), batch.get(keys("mask"))),
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

    def _load_pretrained_weights(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Path not found: {ckpt_path}. Training from scratch.")
            return

        print(f"\n[INFO] Loading pretrained weights from {ckpt_path}")
        try:
            raw_weights = load_peft_weights(ckpt_path, device="cpu")
            current_keys = list(self.model.state_dict().keys())

            def normalize(k):
                ignore = {"base_model", "model", "module", "modules_to_save", "default"}
                return ".".join(p for p in k.split(".") if p not in ignore)

            key_map = {normalize(k): k for k in current_keys}
            final_weights = {
                key_map[normalize(k)]: v
                for k, v in raw_weights.items()
                if normalize(k) in key_map
            }

            if final_weights:
                self.model.load_state_dict(final_weights, strict=False)
                print(f"Transferred {len(final_weights)} pretrained layers.")

            for name, param in self.model.named_parameters():
                if any(t in name for t in [
                    "lora", "modules_to_save", "head",
                    "input_proj", "input_norm", "cross_layers"
                ]):
                    param.requires_grad = True

        except Exception as e:
            print(f"[ERROR] Weight loading failed: {e}")
            raise e
