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
from src.dataset_affinity import ProteinDataModule
from src.finetune_module import ProteinAffinityModule

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
            save_path = os.path.join(self.output_dir, "saved_model")
            print(f"\n[System] New Best Val Loss: {current_val_loss:.4f}. Saving...", flush=True)
            pl_module.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

@hydra.main(version_base=None, config_path="../configs", config_name="affinity")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed, workers=True)
    
    # 1. Get Explicit Absolute Output Path
    hydra_out_dir = HydraConfig.get().runtime.output_dir
    os.environ["WANDB_DIR"] = hydra_out_dir

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[System] Hydra Output Directory: {hydra_out_dir}")
        print(f"[Config] Task: Affinity Fine-tuning")
        
        # Log Logic Status
        if cfg.get("resume_checkpoint_path"):
            print(f"[Config] RESUMING STATE from: {cfg.resume_checkpoint_path}")
        elif cfg.get("pretrained_ckpt_path"):
            print(f"[Config] Transfer Learning (Weights Only): {cfg.pretrained_ckpt_path}")

    # 2. Components
    dm = ProteinDataModule(cfg)
    model = ProteinAffinityModule(cfg)

    # 3. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(hydra_out_dir, "checkpoints"),
        filename="affinity-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True # Useful for resuming later
    )

    hf_save_cb = SaveHuggingFaceFormatCallback(
        output_dir=hydra_out_dir,
        tokenizer=dm.tokenizer
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 4. Logger
    wandb_logger = WandbLogger(
        project=getattr(cfg.training.wandb, "project", "affinity-finetune"),
        save_dir=hydra_out_dir,
        name=f"affinity-{cfg.model.name}",
        log_model=False,
    )

    # 5. Trainer
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
        precision=cfg.training.get("precision", "16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=cfg.training.get("gradient_clip_algorithm", "norm")
    )

    # --- RESUME LOGIC ---
    ckpt_path = cfg.get("resume_checkpoint_path", None)
    
    # Handle absolute path conversion because Hydra changes CWD
    if ckpt_path:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Pass ckpt_path to fit()
    # If ckpt_path is None, it trains from scratch (or transfer weights)
    # If ckpt_path is set, it ignores transfer weights and loads the FULL state
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # 6. Final Export
    if trainer.is_global_zero:
        best_path = checkpoint_cb.best_model_path
        # If we resumed, best path might be from previous run if not improved,
        # so fallback to checkpoint_cb.last_model_path if needed or just skip if empty
        if not best_path and ckpt_path:
             # If we just resumed and finished without improvement, save the resumed state as 'saved_model'
             best_path = ckpt_path

        if best_path:
            print(f"[System] Loading best/last checkpoint: {best_path}")
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            
            final_path = os.path.join(hydra_out_dir, "saved_model")
            print(f"[System] Final Export to: {final_path}")
            model.model.save_pretrained(final_path)
            dm.tokenizer.save_pretrained(final_path)

if __name__ == '__main__':
    main()
