import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics.functional import spearman_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


class BestModelSaver(Callback):
    """
    Saves the best model to 'saved_model/' directory.
    """
    def __init__(self, output_dir, tokenizer, monitor="val_loss", mode="min"):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.mode = mode
        self.save_path = os.path.join(output_dir, "saved_model")

        if mode == "min":
            self.best_value = float('inf')
            self.compare = lambda current, best: current < best
        else:
            self.best_value = float('-inf')
            self.compare = lambda current, best: current > best

        print(f"\n[BestModelSaver] Initialized:")
        print(f"  Monitor: {monitor}")
        print(f"  Mode: {mode}")
        print(f"  Save path: {self.save_path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is None:
            return

        if isinstance(current_value, torch.Tensor):
            current_value = current_value.item()

        if self.compare(current_value, self.best_value):
            self.best_value = current_value
            os.makedirs(self.save_path, exist_ok=True)

            print(f"\n{'='*70}")
            print(f"[BestModelSaver] New best {self.monitor}: {current_value:.4f}")
            print(f"[BestModelSaver] Saving model to: {self.save_path}")
            print(f"{'='*70}\n")

            self._save_model(pl_module)

            metadata = {
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                self.monitor: current_value,
                'mode': self.mode
            }

            metadata_path = os.path.join(self.save_path, "best_model_metadata.txt")
            with open(metadata_path, 'w') as f:
                f.write(f"Best Model Metadata\n{'='*50}\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")

        # Always save the latest model after each validation
        self._save_last_model(trainer, pl_module)

    def _save_last_model(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        last_save_path = os.path.join(self.output_dir, "saved_model_last")
        os.makedirs(last_save_path, exist_ok=True)
        print(f"\n[BestModelSaver] Saving last model to: {last_save_path}")
        self._save_model(pl_module, last_save_path)

    def on_train_end(self, trainer, pl_module):
        self._save_last_model(trainer, pl_module)

    def _save_model(self, pl_module, save_path=None):
        if save_path is None:
            save_path = self.save_path
        try:
            if hasattr(pl_module.model, "save_pretrained"):
                # PEFT/LoRA model
                pl_module.model.save_pretrained(save_path)
            elif hasattr(pl_module.model, "base_model") and hasattr(pl_module.model.base_model, "save_pretrained"):
                pl_module.model.base_model.save_pretrained(save_path)
            else:
                # Baseline (no LoRA): save only trainable custom layer weights
                state_dict = {k: v for k, v in pl_module.model.state_dict().items() if k.startswith(("input_proj", "input_norm", "cross_layers", "pool", "head_affinity"))}
                torch.save(state_dict, os.path.join(save_path, "baseline_model.pt"))
                print(f"[BestModelSaver] Saved baseline custom layers to {save_path}/baseline_model.pt")

            self.tokenizer.save_pretrained(save_path)
            print(f"[BestModelSaver] ✓ Model and tokenizer saved to {save_path}")

        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            raise


class TestEveryValidationCallback(Callback):
    """
    Runs regression test loop every validation (including mid-epoch intervals).
    """
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self._test_dataloader = None

    def on_validation_epoch_end(self, trainer, pl_module):
        # trainer.sanity_checking ensures this doesn't run during the 2-step startup check
        if trainer.sanity_checking:
            return

        if self.datamodule.test_dataset is None:
            return

        if self._test_dataloader is None:
            self._test_dataloader = self.datamodule.test_dataloader()
            if self._test_dataloader is None:
                return

        pl_module.eval()
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch in self._test_dataloader:
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                pred_reg = pl_module.forward(batch)
                reg_labels = batch['reg_labels']

                test_preds.append(pred_reg)
                test_labels.append(reg_labels)

        all_preds = torch.cat(test_preds).reshape(-1)
        all_labels = torch.cat(test_labels).reshape(-1)

        if trainer.world_size > 1:
            all_preds = pl_module.all_gather(all_preds).reshape(-1)
            all_labels = pl_module.all_gather(all_labels).reshape(-1)

        if trainer.is_global_zero:
            avg_loss = torch.nn.functional.mse_loss(all_preds, all_labels)
            spearman = spearman_corrcoef(all_preds, all_labels)

            if trainer.logger:
                trainer.logger.log_metrics({
                    'test_loss': avg_loss.item(),
                    'test_spearman': spearman.item()
                }, step=trainer.global_step)

            print(f"[Test] Step {trainer.global_step} | loss: {avg_loss.item():.4f}, spearman: {spearman.item():.4f}")

        pl_module.train()


class BinaryTestCallback(Callback):
    """
    Evaluates binary classification per target every validation.
    Removed the 'check_every_n_epoch' gate to support val_check_interval.
    """
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self._dataloader = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        # We check if the dataset exists; if so, we run it every time validation ends.
        if self.datamodule.binary_test_dataset is None:
            return

        if self._dataloader is None:
            # Assumes the datamodule handles its own setup or internal dataloader creation
            self._dataloader = self.datamodule.binary_test_dataloader()
            if self._dataloader is None:
                return

        pl_module.eval()
        all_scores = []
        all_labels = []
        all_target_ids = []

        with torch.no_grad():
            for batch in self._dataloader:
                batch_device = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}

                scores = pl_module.forward(batch_device)

                all_scores.extend(scores.cpu().flatten().tolist())
                all_labels.extend(batch['is_binder'].tolist())
                all_target_ids.extend(batch['tid'])

        # Gather across GPUs for DDP support
        if trainer.world_size > 1:
            gathered_scores = [None] * trainer.world_size
            gathered_labels = [None] * trainer.world_size
            gathered_tids = [None] * trainer.world_size

            torch.distributed.all_gather_object(gathered_scores, all_scores)
            torch.distributed.all_gather_object(gathered_labels, all_labels)
            torch.distributed.all_gather_object(gathered_tids, all_target_ids)

            if trainer.is_global_zero:
                all_scores = [s for sublist in gathered_scores for s in sublist]
                all_labels = [l for sublist in gathered_labels for l in sublist]
                all_target_ids = [t for sublist in gathered_tids for t in sublist]

        if trainer.is_global_zero:
            self._compute_and_log_metrics(trainer, all_scores, all_labels, all_target_ids)

        pl_module.train()

    def _compute_and_log_metrics(self, trainer, all_scores, all_labels, all_target_ids):
        df = pd.DataFrame({
            'score': all_scores,
            'is_binder': all_labels,
            'target_id': all_target_ids
        })

        target_aucs, target_aps, target_accs = [], [], []

        for tid, group in df.groupby('target_id'):
            if len(group) < 2 or group['is_binder'].nunique() < 2:
                continue

            neg_scores = -np.array(group['score'].values)  # Negate: lower raw score = better binder
            labels = group['is_binder'].values

            try:
                target_aucs.append(roc_auc_score(labels, neg_scores))
                target_aps.append(average_precision_score(labels, neg_scores))
                preds = (neg_scores > np.median(neg_scores)).astype(int)
                target_accs.append(accuracy_score(labels, preds))
            except Exception:
                continue

        if target_aucs:
            metrics = {
                'binary_test/avg_auc': np.mean(target_aucs),
                'binary_test/avg_ap': np.mean(target_aps),
                'binary_test/avg_acc': np.mean(target_accs),
                'binary_test/num_targets': len(target_aucs),
            }
            if trainer.logger:
                trainer.logger.log_metrics(metrics, step=trainer.global_step)

            print(f"[BinaryTest] Step {trainer.global_step} | Avg AUC: {metrics['binary_test/avg_auc']:.4f}")
