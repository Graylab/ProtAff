import sys
import os
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig  
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset_dms import DMSDataModule
from src.pretrain_module import DMSModule

class SaveHuggingFaceFormatCallback(Callback):
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.best_val_loss = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        
        current_val_loss = trainer.callback_metrics.get("val_loss")
        if current_val_loss is None: return

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            # Save strictly inside the Hydra output directory
            save_path = os.path.join(self.output_dir, "saved_model")
            print(f"\n[System] New Best Val Loss: {current_val_loss:.4f}. Saving...", flush=True)
            pl_module.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

@hydra.main(version_base=None, config_path="../configs", config_name="dms")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed, workers=True)
    
    # 1. Get Explicit Absolute Output Path from Hydra
    hydra_out_dir = HydraConfig.get().runtime.output_dir
    
    # 2. Force WandB to save inside this directory
    os.environ["WANDB_DIR"] = hydra_out_dir
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[System] Hydra Output Directory: {hydra_out_dir}")
        print(f"[Config] Task: DMS Pre-training")
        
        # Log Logic Status
        if cfg.get("resume_checkpoint_path"):
            print(f"[Config] RESUMING STATE from: {cfg.resume_checkpoint_path}")
        else:
            print(f"[Config] Training from Scratch")

    # 3. Init Data & Model
    dm = DMSDataModule(cfg)
    model = DMSModule(cfg)

    # 4. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(hydra_out_dir, "checkpoints"),
        filename="dms-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True # Essential for resuming
    )

    hf_save_cb = SaveHuggingFaceFormatCallback(
        output_dir=hydra_out_dir,
        tokenizer=dm.tokenizer
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 5. Logger
    wandb_logger = WandbLogger(
        project=getattr(cfg.training.wandb, "project", "dms-pretrain"),
        save_dir=hydra_out_dir, 
        name=f"dms-{cfg.model.name}",
        log_model=False,
        # Optional: Allow resuming WandB plots if ID provided
        # id=cfg.training.wandb.get("id", None), 
        # resume="allow"
    )

    # 6. Trainer
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.get("strategy", "auto"),
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, hf_save_cb, lr_monitor],
        default_root_dir=hydra_out_dir, 
        log_every_n_steps=10,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        precision="bf16-mixed" if cfg.training.get("use_mixed_precision", False) else 32,
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=cfg.training.get("gradient_clip_algorithm", "norm")
    )

    # --- RESUME LOGIC ---
    ckpt_path = cfg.get("resume_checkpoint_path", None)
    
    if ckpt_path:
        # Convert relative path to absolute because Hydra changes CWD
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 7. Start Training (With Resume Support)
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # 8. Final Export
    if trainer.is_global_zero:
        best_path = checkpoint_cb.best_model_path
        
        # Fallback: if we just resumed and finished without a new "best", use the resume path
        if not best_path and ckpt_path:
            best_path = ckpt_path

        if best_path:
            print(f"[System] Loading checkpoint for export: {best_path}")
            # Load weights to CPU to avoid memory issues during export
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            
            final_path = os.path.join(hydra_out_dir, "saved_model")
            print(f"[System] Final Export to: {final_path}")
            model.model.save_pretrained(final_path)
            dm.tokenizer.save_pretrained(final_path)

if __name__ == '__main__':
    main()