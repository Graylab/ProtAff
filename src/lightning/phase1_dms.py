import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup
from src.models import build_model 

class DMSModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Container (Factory: returns ESMCrossAttentionClassifier or ESMConcatModel)
        self.base_model = build_model(cfg)
        
        # 2. Setup PEFT (Conditional on Architecture)
        arch = getattr(cfg.model, "arch", "cross_attn")

        if arch == "concat":
            # Concat Model: Only Projector and Head need saving
            # No Decoders or Poolers exist in this architecture
            custom_modules = ["projector", "norm_input", "head_score"]
        else:
            # Dual/Siamese Model: Save everything
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
            if mod not in modules_to_save:
                modules_to_save.append(mod)

        peft_config = LoraConfig(
            r=cfg.model.lora.r, 
            lora_alpha=cfg.model.lora.alpha, 
            target_modules=target_modules,
            modules_to_save=modules_to_save, 
            lora_dropout=cfg.model.lora.dropout, 
            bias="none"
        )

        self.model = get_peft_model(self.base_model, peft_config)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, batch):
        # We access base_model directly to ensure argument mapping works
        preds = self.model.base_model(
            binder_ids=batch['mut_ids'], 
            binder_mask=batch['mut_mask'],
            target_ids=batch['wt_ids'], 
            target_mask=batch['wt_mask']
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