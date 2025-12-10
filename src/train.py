import sys
import os
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig  
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import your modules
from src.datasets.dataset_dms import DMSDataModule
from src.datasets.dataset_affinity import AffinityDataModule 
from src.lightning.phase1_dms import DMSModule        
from src.lightning.phase2_affinity import AffinityModule

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
            # Access the underlying PEFT model
            if hasattr(pl_module.model, "save_pretrained"):
                pl_module.model.save_pretrained(save_path)
            else:
                # Fallback if structure is nested
                pl_module.model.base_model.save_pretrained(save_path)
                
            self.tokenizer.save_pretrained(save_path)

def get_task_modules(cfg):
    """
    Factory to select the correct DataModule and LightningModule
    based on the 'task_name' in config.
    """
    task = cfg.get("task_name", "dms") # Default to DMS if not set
    
    if task == "dms":
        # Phase 1: Mutant vs Wildtype
        dm = DMSDataModule(cfg)
        model = DMSModule(cfg)
    elif task == "affinity":
        # Phase 2: Binder vs Target
        dm = AffinityDataModule(cfg)
        model = AffinityModule(cfg)
    else:
        raise ValueError(f"Unknown task_name: {task}. Options: ['dms', 'affinity']")
    
    return dm, model

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed, workers=True)
    
    # 1. Get Explicit Absolute Output Path from Hydra
    hydra_out_dir = HydraConfig.get().runtime.output_dir
    
    # 2. Force WandB to save inside this directory
    os.environ["WANDB_DIR"] = hydra_out_dir
    
    # Determine Task
    task_name = cfg.get("task_name", "dms")

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[System] Hydra Output Directory: {hydra_out_dir}")
        print(f"[Config] Task: {task_name.upper()}")
        
        # Log Logic Status
        if cfg.get("resume_checkpoint_path"):
            print(f"[Config] RESUMING STATE from: {cfg.resume_checkpoint_path}")
        elif cfg.get("pretrained_ckpt_path"):
             print(f"[Config] TRANSFER LEARNING from: {cfg.pretrained_ckpt_path}")
        else:
            print(f"[Config] Training from Scratch")

    # 3. Init Data & Model (Dynamic)
    dm, model = get_task_modules(cfg)

    # 4. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(hydra_out_dir, "checkpoints"),
        filename=f"{task_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True 
    )

    hf_save_cb = SaveHuggingFaceFormatCallback(
        output_dir=hydra_out_dir,
        tokenizer=dm.tokenizer
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks_list = [checkpoint_cb, hf_save_cb, lr_monitor]

    # Add EarlyStopping if defined in config (Crucial for Phase 2)
    if cfg.training.get("early_stopping", False):
        es_patience = cfg.training.get("patience", 3)
        callbacks_list.append(EarlyStopping(monitor="val_loss", patience=es_patience, mode="min"))

    # 5. Logger
    wandb_logger = WandbLogger(
        project=getattr(cfg.training.wandb, "project", f"{task_name}-project"),
        save_dir=hydra_out_dir, 
        name=f"{task_name}-{cfg.model.name}",
        log_model=False,
    )

    # 6. Trainer
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.get("strategy", "auto"),
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks_list,
        default_root_dir=hydra_out_dir, 
        log_every_n_steps=10,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        precision=cfg.training.get("precision", "16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=cfg.training.get("gradient_clip_algorithm", "norm"),
        # Optimization for large datasets
        limit_val_batches=cfg.training.get("limit_val_batches", None), 
        val_check_interval=cfg.training.get("val_check_interval", 1.0)
    )

    # --- RESUME LOGIC (For recovering crashed runs) ---
    ckpt_path = cfg.get("resume_checkpoint_path", None)
    
    if ckpt_path:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 7. Start Training
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # 8. Final Export
    if trainer.is_global_zero:
        best_path = checkpoint_cb.best_model_path
        
        # Fallback logic
        if not best_path and ckpt_path:
            best_path = ckpt_path

        if best_path:
            print(f"[System] Loading best checkpoint for export: {best_path}")
            # Load weights to CPU
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            
            final_path = os.path.join(hydra_out_dir, "saved_model")
            print(f"[System] Final Export to: {final_path}")
            
            # Save PEFT model
            model.model.save_pretrained(final_path)
            dm.tokenizer.save_pretrained(final_path)

if __name__ == '__main__':
    main()
