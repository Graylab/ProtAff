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
        
        # 1. Base Container
        self.base_model = build_model(cfg)
        
        # 2. Setup PEFT (LoRA)
        # We focus on the modules: Projector + Head (Score only)
        custom_modules = ["projector", "norm_input", "head_score"]
        
        # Combine with config-defined modules
        modules_to_save = OmegaConf.to_container(cfg.model.lora.modules_to_save) if cfg.model.lora.modules_to_save else []
        for mod in custom_modules:
            if mod not in modules_to_save:
                modules_to_save.append(mod)

        target_modules = OmegaConf.to_container(cfg.model.lora.target_modules)

        peft_config = LoraConfig(
            r=cfg.model.lora.r, 
            lora_alpha=cfg.model.lora.alpha, 
            target_modules=target_modules,
            modules_to_save=modules_to_save, 
            lora_dropout=cfg.model.lora.dropout, 
            bias="none"
        )

        self.model = get_peft_model(self.base_model, peft_config)
        
        # 3. Loss Functions
        # Pure Regression: Measures how close the predicted stability score is to the DMS truth
        self.mse_loss = nn.MSELoss()
    
    def forward(self, input_ids, attention_mask):
        """
        Standard forward pass. 
        Matches Phase 2 signature for easy transfer.
        """
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ) 

    def training_step(self, batch, batch_idx):
        # 1. Forward
        pred_reg = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask']
        )
        
        # 2. Unpack Labels
        reg_labels = batch['reg_labels'] # Shape: [Batch, 1]
        
        # 3. Compute Loss (MSE)
        loss = self.mse_loss(pred_reg, reg_labels)
        
        # 4. Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        pred_reg = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask']
        )
        
        reg_labels = batch['reg_labels']
        
        loss = self.mse_loss(pred_reg, reg_labels)
        
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