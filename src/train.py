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

# --- IMPORT MODULES ---
# Phase 1: DMS
from src.datasets.dataset_dms import DMSDataModule
from src.lightning.phase1_dms import DMSModule

# Phase 2a: Affinity (Regression)
from src.datasets.dataset_affinity import AffinityDataModule 
from src.lightning.phase2_affinity import AffinityModule

# Phase 2b: Pair Affinity (Ranking)
from src.datasets.dataset_pair_affinity import PairAffinityDataModule
from src.lightning.phase2_pair_affinity import PairAffinityModule


class SaveHuggingFaceFormatCallback(Callback):
    """
    Saves the PEFT model directly to 'saved_model/' whenever a new best validation loss is achieved.
    """
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.best_val_loss = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        # --- MINIMAL CHANGE: Prevent saving during sanity check ---
        if not trainer.is_global_zero or trainer.sanity_checking: 
            return
        
        current_val_loss = trainer.callback_metrics.get("val_loss")
        if current_val_loss is None: return

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            
            # CHANGED: Save directly to "saved_model" (no _best suffix)
            save_path = os.path.join(self.output_dir, "saved_model")
            os.makedirs(save_path, exist_ok=True)

            print(f"\n[System] New Best Val Loss: {current_val_loss:.4f}. Updating 'saved_model'...", flush=True)
            
            # Robust saving for nested PEFT models
            if hasattr(pl_module.model, "save_pretrained"):
                pl_module.model.save_pretrained(save_path)
            elif hasattr(pl_module.model, "base_model") and hasattr(pl_module.model.base_model, "save_pretrained"):
                pl_module.model.base_model.save_pretrained(save_path)
            else:
                print("[WARN] Could not find 'save_pretrained' method on model wrapper.")
                
            self.tokenizer.save_pretrained(save_path)


def get_task_modules(cfg):
    """
    Factory: Returns (DataModule, LightningModule) based on cfg.task_name
    """
    task = cfg.get("task_name", "dms").lower()
    
    if task == "dms":
        print("[Factory] Loading Phase 1: DMS (Regression/Classification)")
        dm = DMSDataModule(cfg)
        model = DMSModule(cfg)
        
    elif task == "affinity":
        print("[Factory] Loading Phase 2a: Affinity (Regression)")
        dm = AffinityDataModule(cfg)
        model = AffinityModule(cfg)
        
    elif task == "pair_affinity":
        print("[Factory] Loading Phase 2b: Pair Affinity (Ranking)")
        # Using the classes from your snippet
        dm = PairAffinityDataModule(cfg)
        model = PairAffinityModule(cfg)
        
    else:
        raise ValueError(f"Unknown task_name: {task}. Options: ['dms', 'affinity', 'pair_affinity']")
    
    return dm, model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Reproducibility
    pl.seed_everything(cfg.training.seed, workers=True)
    
    # 2. Output Paths
    hydra_out_dir = HydraConfig.get().runtime.output_dir
    os.environ["WANDB_DIR"] = hydra_out_dir # Force WandB local logs here
    
    # 3. Initialize Task Modules
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[System] Hydra Output: {hydra_out_dir}")
        
    dm, model = get_task_modules(cfg)

    # 4. Define Callbacks
    # Filename includes accuracy if available (useful for ranking)
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(hydra_out_dir, "checkpoints"),
        filename=f"{cfg.task_name}-{{epoch:02d}}-{{val_loss:.4f}}",
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

    # Early Stopping (Optional but recommended)
    if cfg.training.get("early_stopping", False):
        patience = cfg.training.get("patience", 3)
        callbacks_list.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))

    # 5. Logger
    wandb_logger = WandbLogger(
        project=getattr(cfg.training.wandb, "project", f"{cfg.task_name}-project"),
        save_dir=hydra_out_dir, 
        name=f"{cfg.task_name}-{cfg.model.name}",
        log_model=False,
    )

    # 6. Trainer Configuration
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.get("strategy", "auto"),
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks_list,
        default_root_dir=hydra_out_dir,
        
        # Optimization Params
        log_every_n_steps=cfg.training.get("log_every_n_steps", 10),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        precision=cfg.training.get("precision", "16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        
        # Validation Frequency (useful for Ranking which is slow)
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
        limit_val_batches=cfg.training.get("limit_val_batches", None)
    )

    # 7. Checkpoint / Resume Logic
    ckpt_path = cfg.get("resume_checkpoint_path", None)
    if ckpt_path:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        print(f"[System] Resuming training from: {ckpt_path}")

    # 8. Start Training
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # 9. Final Verification Export
    if trainer.is_global_zero:
        best_path = checkpoint_cb.best_model_path
        
        if not best_path and ckpt_path:
            best_path = ckpt_path

        if best_path:
            print(f"\n[System] Loading best checkpoint for verification: {best_path}")
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            
            # CHANGED: Unified path "saved_model" (no _final suffix)
            final_path = os.path.join(hydra_out_dir, "saved_model")
            print(f"[System] Final Verification Export to: {final_path}")
            
            model.model.save_pretrained(final_path)
            dm.tokenizer.save_pretrained(final_path)

if __name__ == '__main__':
    main()