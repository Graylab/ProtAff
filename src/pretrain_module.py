import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
from omegaconf import DictConfig, OmegaConf
from src.model_base import ESMCrossAttentionClassifier

class DMSModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # 1. Base Container
        self.base_model = ESMCrossAttentionClassifier(cfg.model.name, cfg=cfg)
        
        # 2. Inject Phase 1 Specific Head (Single Sequence)
        d_model = getattr(cfg.model, "d_model", 128)
        self.base_model.single_head = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.ReLU(), 
            nn.Linear(d_model, 1)
        )

        # 3. Setup LoRA (Only saving Encoder parts)
        custom_modules = ["projector", "norm_input", "task_encoder", "norm_pooled_b", "single_head"]
        if getattr(cfg.model, "pooling", "mean") == "attention":
            custom_modules.append("pooler_b")

        target_modules = OmegaConf.to_container(cfg.model.lora.target_modules)
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

    # --- PHASE 1 FORWARD ---
    def forward_single(self, ids, mask):
        bm = self.model.base_model.model
        
        # 1. ESM
        raw = bm.esm(ids, attention_mask=mask).last_hidden_state
        # 2. Project
        vec = bm.norm_input(bm.projector(raw))
        # 3. Encoder (Training this sensitivity!)
        enc = bm.task_encoder(vec, src_key_padding_mask=~mask.bool())
        
        # 4. Pool (Training this attention!)
        # Note: We pool the ENCODER output directly
        pool_mask = bm.get_pooling_mask(ids, mask)
        
        if bm.pooling_type == "attention":
            pooled = bm.pooler_b(enc, pool_mask)
        else:
            pooled = (enc * pool_mask).sum(dim=1) / torch.clamp(pool_mask.sum(dim=1), min=1e-9)
        
        pooled = bm.norm_pooled_b(pooled)
        return bm.single_head(pooled)

    def training_step(self, batch, batch_idx):
        pred_wt = self.forward_single(batch['wt_ids'], batch['wt_mask'])
        pred_mut = self.forward_single(batch['mut_ids'], batch['mut_mask'])
        
        # MSE on Delta
        loss = self.loss_fn(pred_mut - pred_wt, batch['labels'].unsqueeze(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_wt = self.forward_single(batch['wt_ids'], batch['wt_mask'])
        pred_mut = self.forward_single(batch['mut_ids'], batch['mut_mask'])
        loss = self.loss_fn(pred_mut - pred_wt, batch['labels'].unsqueeze(-1))
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.training.learning_rate)