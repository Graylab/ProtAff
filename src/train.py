import sys
import os
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig  
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import spearman_corrcoef
            

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# --- IMPORT MODULES ---
# Phase 1: DMS (Regression)
from src.datasets.dataset_dms import DMSDataModule
from src.lightning.phase1_dms import DMSModule

# Phase 1: DMS (Ranking)
from src.datasets.dataset_pair_dms import PairDMSDataModule
from src.lightning.phase1_pair_dms import PairDMSModule

# Phase 2: Affinity (Regression)
from src.datasets.dataset_affinity import AffinityDataModule 
from src.lightning.phase2_affinity import AffinityModule

# Phase 2: Pair Affinity (Ranking)
from src.datasets.dataset_pair_affinity import PairAffinityDataModule
from src.lightning.phase2_pair_affinity import PairAffinityModule


class BestModelSaver(Callback):
    """
    Unified callback that saves the best model to 'saved_model/' directory.
    Monitors any metric (val_loss, val_spearman, etc.) with flexible mode (min/max).
    
    This is the MOST IMPORTANT callback - ensures we always have the best model saved.
    """
    def __init__(self, output_dir, tokenizer, monitor="val_loss", mode="min"):
        """
        Args:
            output_dir: Base output directory (will save to output_dir/saved_model/)
            tokenizer: Tokenizer to save alongside model
            monitor: Metric to monitor (e.g., 'val_loss', 'val_spearman')
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.mode = mode
        self.save_path = os.path.join(output_dir, "saved_model")
        
        # Track best value
        if mode == "min":
            self.best_value = float('inf')
            self.compare = lambda current, best: current < best
        else:  # mode == "max"
            self.best_value = float('-inf')
            self.compare = lambda current, best: current > best
        
        print(f"\n[BestModelSaver] Initialized:")
        print(f"  Monitor: {monitor}")
        print(f"  Mode: {mode}")
        print(f"  Save path: {self.save_path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking: 
            return
        
        # Get current metric value
        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is None:
            return
        
        # Convert tensor to float if needed
        if isinstance(current_value, torch.Tensor):
            current_value = current_value.item()

        # Check if this is the best value so far
        if self.compare(current_value, self.best_value):
            self.best_value = current_value
            
            os.makedirs(self.save_path, exist_ok=True)
            
            print(f"\n{'='*70}")
            print(f"[BestModelSaver] New best {self.monitor}: {current_value:.4f}")
            print(f"[BestModelSaver] Saving model to: {self.save_path}")
            print(f"{'='*70}\n")
            
            # Save model
            self._save_model(pl_module)
            
            # Save metadata about this checkpoint
            metadata = {
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                self.monitor: current_value,
                'mode': self.mode
            }
            
            metadata_path = os.path.join(self.save_path, "best_model_metadata.txt")
            with open(metadata_path, 'w') as f:
                f.write(f"Best Model Metadata\n")
                f.write(f"{'='*50}\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"[BestModelSaver] Metadata saved to: {metadata_path}\n")
    
    def _save_model(self, pl_module):
        """Save the PEFT model and tokenizer"""
        try:
            # Try to save PEFT model directly
            if hasattr(pl_module.model, "save_pretrained"):
                pl_module.model.save_pretrained(self.save_path)
            elif hasattr(pl_module.model, "base_model") and hasattr(pl_module.model.base_model, "save_pretrained"):
                pl_module.model.base_model.save_pretrained(self.save_path)
            else:
                print("[WARN] Could not find 'save_pretrained' method on model.")
                return
            
            # Save tokenizer
            self.tokenizer.save_pretrained(self.save_path)
            
            print(f"[BestModelSaver] ✓ Model and tokenizer saved successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            raise


class TestEveryValidationCallback(Callback):
    """
    Runs test loop every time validation is run.
    Properly handles DDP by gathering results across all GPUs.
    """
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self._test_dataloader = None

    def on_validation_end(self, trainer, pl_module):
        # Skip sanity check
        if trainer.sanity_checking:
            return
        
        # Skip if no test dataset
        if self.datamodule.test_dataset is None:
            return
        
        # Lazy init test dataloader
        if self._test_dataloader is None:
            self.datamodule.setup("test")
            self._test_dataloader = self.datamodule.test_dataloader()
            if self._test_dataloader is None:
                return
        
        # Manually run test loop
        pl_module.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in self._test_dataloader:
                # Move batch to device
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                pred_reg = pl_module.forward(batch)
                reg_labels = batch['reg_labels']
                
                test_preds.append(pred_reg)
                test_labels.append(reg_labels)
        
        # Concatenate local results
        all_preds = torch.cat(test_preds).reshape(-1)
        all_labels = torch.cat(test_labels).reshape(-1)
        
        # Gather across GPUs if using DDP
        if trainer.world_size > 1:
            all_preds = pl_module.all_gather(all_preds).reshape(-1)
            all_labels = pl_module.all_gather(all_labels).reshape(-1)
        
        # Compute metrics (only on rank 0 to avoid duplicate logging)
        if trainer.is_global_zero:
            avg_loss = torch.nn.functional.mse_loss(all_preds, all_labels)
            spearman = spearman_corrcoef(all_preds, all_labels)
            
            # Log directly to logger
            if trainer.logger:
                trainer.logger.log_metrics({
                    'test_loss': avg_loss.item(),
                    'test_spearman': spearman.item()
                }, step=trainer.global_step)
                
            print(f"[Test] loss: {avg_loss.item():.4f}, spearman: {spearman.item():.4f}")
        
        pl_module.train()


def get_task_modules(cfg):
    """
    Factory: Returns (DataModule, LightningModule) based on cfg.task_name
    """
    task = cfg.get("task_name", "dms").lower()
    
    if task == "dms":
        print("[Factory] Loading Phase 1: DMS (Regression)")
        dm = DMSDataModule(cfg)
        model = DMSModule(cfg)

    elif task == "pair_dms":
        print("[Factory] Loading Phase 1: DMS (Pair Ranking)")
        dm = PairDMSDataModule(cfg)
        model = PairDMSModule(cfg)
        
    elif task == "affinity":
        print("[Factory] Loading Phase 2: Affinity (Regression)")
        dm = AffinityDataModule(cfg)
        model = AffinityModule(cfg)
        
    elif task == "pair_affinity":
        print("[Factory] Loading Phase 2: Pair Affinity (Ranking)")
        dm = PairAffinityDataModule(cfg)
        model = PairAffinityModule(cfg)
        
    else:
        raise ValueError(f"Unknown task_name: {task}. Options: ['dms', 'pair_dms', 'affinity', 'pair_affinity']")
    
    return dm, model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # ======================================================================
    # SETUP
    # ======================================================================
    
    # 1. Reproducibility
    pl.seed_everything(cfg.training.seed, workers=True)
    
    # 2. Output Paths
    hydra_out_dir = HydraConfig.get().runtime.output_dir
    os.environ["WANDB_DIR"] = hydra_out_dir
    
    # 3. Initialize Task Modules
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n{'='*70}")
        print(f"[System] Starting Training")
        print(f"{'='*70}")
        print(f"[System] Hydra Output: {hydra_out_dir}")
        print(f"[System] Task: {cfg.task_name}")
        
    dm, model = get_task_modules(cfg)

    # ======================================================================
    # CALLBACKS CONFIGURATION
    # ======================================================================
    
    callbacks_list = []
    
    # 1. MOST IMPORTANT: Best Model Saver
    # This saves the final model that will be used in production
    monitor_metric = cfg.training.get("monitor_metric", "val_loss")
    monitor_mode = cfg.training.get("monitor_mode", "min")
    
    best_model_saver = BestModelSaver(
        output_dir=hydra_out_dir,
        tokenizer=dm.tokenizer,
        monitor=monitor_metric,
        mode=monitor_mode
    )
    callbacks_list.append(best_model_saver)
    
    # 2. Standard PyTorch Lightning Checkpoint (for resuming training)
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(hydra_out_dir, "checkpoints"),
        filename=f"{cfg.task_name}-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        save_top_k=3,                    # Keep top 3 checkpoints
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,                  # Always save last checkpoint for resuming
        verbose=True
    )
    callbacks_list.append(checkpoint_cb)
    
    # 3. Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_list.append(lr_monitor)

    # 4. Test Every Validation (optional, for monitoring generalization)
    if cfg.training.get("test_every_val", False):
        test_cb = TestEveryValidationCallback(datamodule=dm)
        callbacks_list.append(test_cb)
        print("[System] Test loop will run after each validation")

    # 5. Early Stopping (optional)
    if cfg.training.get("early_stopping", False):
        patience = cfg.training.get("early_stopping_patience", 5)
        early_stop_cb = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode=monitor_mode,
            verbose=True
        )
        callbacks_list.append(early_stop_cb)
        print(f"[System] Early stopping enabled (patience={patience})")

    # ======================================================================
    # LOGGER
    # ======================================================================
    
    wandb_logger = WandbLogger(
        project=cfg.training.wandb.get("project", f"{cfg.task_name}-project"),
        name=cfg.training.wandb.get("name", f"{cfg.task_name}-{cfg.model.name}"),
        save_dir=hydra_out_dir, 
        log_model=False,
        config=dict(cfg)  # Log full config to wandb
    )

    # ======================================================================
    # TRAINER
    # ======================================================================
    
    trainer = pl.Trainer(
        # Hardware
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.get("strategy", "auto"),
        
        # Training
        max_epochs=cfg.training.max_epochs,
        
        # Logging & Callbacks
        logger=wandb_logger,
        callbacks=callbacks_list,
        default_root_dir=hydra_out_dir,
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        
        # Optimization
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        precision=cfg.training.get("precision", "16-mixed"),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        
        # Validation
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
        limit_val_batches=cfg.training.get("limit_val_batches", None),
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1)
    )

    # ======================================================================
    # RESUME FROM CHECKPOINT (if specified)
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
    # FINAL MODEL VERIFICATION & EXPORT
    # ======================================================================
    
    if trainer.is_global_zero:
        print(f"\n{'='*70}")
        print(f"[System] Training Complete - Final Model Export")
        print(f"{'='*70}")
        
        final_model_path = os.path.join(hydra_out_dir, "saved_model")
        
        # Check if BestModelSaver already saved the model
        if os.path.exists(os.path.join(final_model_path, "adapter_model.safetensors")) or \
           os.path.exists(os.path.join(final_model_path, "adapter_model.bin")):
            print(f"\n✓ Best model already saved by BestModelSaver")
            print(f"  Location: {final_model_path}")
            print(f"  Best {monitor_metric}: {best_model_saver.best_value:.4f}")
            
            # Print what's in the directory
            print(f"\n[System] Final model directory contents:")
            for item in os.listdir(final_model_path):
                print(f"  - {item}")
        else:
            print(f"\n[WARN] No model found in {final_model_path}")
            print(f"[System] Attempting to load and save best checkpoint...")
            
            # Fallback: load best checkpoint and save
            best_path = checkpoint_cb.best_model_path
            if not best_path and ckpt_path:
                best_path = ckpt_path
            
            if best_path and os.path.exists(best_path):
                print(f"[System] Loading best checkpoint: {best_path}")
                ckpt = torch.load(best_path, map_location="cpu")
                model.load_state_dict(ckpt["state_dict"])
                
                os.makedirs(final_model_path, exist_ok=True)
                model.model.save_pretrained(final_model_path)
                dm.tokenizer.save_pretrained(final_model_path)
                
                print(f"✓ Model saved to: {final_model_path}")
            else:
                print(f"[ERROR] Could not find checkpoint to save!")
        
        print(f"\n{'='*70}")
        print(f"[System] All Done!")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()