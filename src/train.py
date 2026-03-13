import sys
import os
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# --- IMPORT MODULES ---
# DataModules
from src.datasets.dataset_affinity import AffinityDataModule
from src.datasets.dataset_pair_affinity import PairAffinityDataModule

# Unified Lightning Modules
from src.lightning.regression_module import RegressionModule
from src.lightning.ranking_module import RankingModule

# Callbacks
from src.callbacks import BestModelSaver, TestEveryValidationCallback, BinaryTestCallback


TASK_MAP = {
    "affinity":      (AffinityDataModule,     RegressionModule),
    "pair_affinity": (PairAffinityDataModule, RankingModule),
}


def get_task_modules(cfg):
    """Factory: Returns (DataModule, LightningModule) based on cfg.task_name"""
    task = cfg.get("task_name", "affinity").lower()

    if task not in TASK_MAP:
        raise ValueError(f"Unknown task_name: {task}. Options: {list(TASK_MAP.keys())}")

    dm_cls, model_cls = TASK_MAP[task]
    task_type = "Ranking" if "pair" in task else "Regression"
    print(f"[Factory] Loading {task} ({task_type})")

    return dm_cls(cfg), model_cls(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # ======================================================================
    # SETUP
    # ======================================================================
    pl.seed_everything(cfg.training.seed, workers=True)
    
    hydra_out_dir = HydraConfig.get().runtime.output_dir
    os.environ["WANDB_DIR"] = hydra_out_dir
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n{'='*70}")
        print(f"[System] Starting Training")
        print(f"{'='*70}")
        print(f"[System] Hydra Output: {hydra_out_dir}")
        print(f"[System] Task: {cfg.task_name}")
        
    dm, model = get_task_modules(cfg)

    # ======================================================================
    # CALLBACKS
    # ======================================================================
    callbacks_list = []
    
    monitor_metric = cfg.training.get("monitor_metric", "val_loss")
    monitor_mode = cfg.training.get("monitor_mode", "min")
    
    # 1. Best Model Saver
    best_model_saver = BestModelSaver(
        output_dir=hydra_out_dir,
        tokenizer=dm.tokenizer,
        monitor=monitor_metric,
        mode=monitor_mode
    )
    callbacks_list.append(best_model_saver)
    
    # 2. Checkpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(hydra_out_dir, "checkpoints"),
        filename=f"{cfg.task_name}-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        save_top_k=0,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        verbose=True
    )
    callbacks_list.append(checkpoint_cb)
    
    # 3. LR Monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_list.append(lr_monitor)

    # 4. Regression Test Callback
    if cfg.training.get("test_every_val", False):
        test_cb = TestEveryValidationCallback(datamodule=dm)
        callbacks_list.append(test_cb)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("[System] Regression test will run after each validation")

    # 5. Binary Test Callback
    if cfg.data.get("binary_test_csv") and cfg.training.get("test_binary_every_val", False):
        binary_test_cb = BinaryTestCallback(datamodule=dm)
        callbacks_list.append(binary_test_cb)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("[System] Binary classification test will run after validation")

    # 6. Early Stopping
    if cfg.training.get("early_stopping", False):
        patience = cfg.training.get("early_stopping_patience", 5)
        early_stop_cb = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode=monitor_mode,
            verbose=True
        )
        callbacks_list.append(early_stop_cb)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[System] Early stopping enabled (patience={patience})")

    # ======================================================================
    # LOGGER
    # ======================================================================
    wandb_logger = WandbLogger(
        project=cfg.training.wandb.get("project", f"{cfg.task_name}-project"),
        name=cfg.training.wandb.get("name", f"{cfg.task_name}-{cfg.model.name}"),
        save_dir=hydra_out_dir, 
        log_model=False,
        config=dict(cfg)
    )

    # ======================================================================
    # TRAINER
    # ======================================================================
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.get("strategy", "auto"),
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks_list,
        default_root_dir=hydra_out_dir,
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        precision=cfg.training.get("precision", "16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
        limit_val_batches=cfg.training.get("limit_val_batches", None),
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1),
        reload_dataloaders_every_n_epochs=1 if cfg.data.get("resample_each_epoch", False) else 0
    )

    # ======================================================================
    # RESUME
    # ======================================================================
    ckpt_path = cfg.get("resume_checkpoint_path", None)
    if ckpt_path:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        print(f"\n[System] Resuming training from: {ckpt_path}\n")

    # ======================================================================
    # TRAIN
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"[System] Starting Training Loop")
    print(f"{'='*70}\n")
    
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # ======================================================================
    # FINAL EXPORT
    # ======================================================================
    if trainer.is_global_zero:
        print(f"\n{'='*70}")
        print(f"[System] Training Complete - Final Model Export")
        print(f"{'='*70}")
        
        final_model_path = os.path.join(hydra_out_dir, "saved_model")
        
        if os.path.exists(os.path.join(final_model_path, "adapter_model.safetensors")) or \
           os.path.exists(os.path.join(final_model_path, "adapter_model.bin")) or \
           os.path.exists(os.path.join(final_model_path, "baseline_model.pt")):
            print(f"\n✓ Best model already saved by BestModelSaver")
            print(f"  Location: {final_model_path}")
            print(f"  Best {monitor_metric}: {best_model_saver.best_value:.4f}")
            
            print(f"\n[System] Final model directory contents:")
            for item in os.listdir(final_model_path):
                print(f"  - {item}")
        else:
            print(f"\n[WARN] No model found in {final_model_path}")
            print(f"[System] Attempting to load and save best checkpoint...")
            
            best_path = checkpoint_cb.best_model_path
            if not best_path and ckpt_path:
                best_path = ckpt_path
            
            if best_path and os.path.exists(best_path):
                print(f"[System] Loading best checkpoint: {best_path}")
                ckpt = torch.load(best_path, map_location="cpu")
                model.load_state_dict(ckpt["state_dict"])
                
                os.makedirs(final_model_path, exist_ok=True)
                if hasattr(model.model, "save_pretrained"):
                    model.model.save_pretrained(final_model_path)
                else:
                    state_dict = {k: v for k, v in model.model.state_dict().items() if k.startswith(("input_proj", "input_norm", "cross_layers", "pool", "head_affinity"))}
                    torch.save(state_dict, os.path.join(final_model_path, "baseline_model.pt"))
                dm.tokenizer.save_pretrained(final_model_path)
                
                print(f"✓ Model saved to: {final_model_path}")
            else:
                print(f"[ERROR] Could not find checkpoint to save!")
        
        print(f"\n{'='*70}")
        print(f"[System] All Done!")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()