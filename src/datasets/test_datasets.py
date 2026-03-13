import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset


class TestRegressionDataset(Dataset):
    """
    Test dataset for regression evaluation.
    Expected CSV format: binder_sequence, target_sequence, log_Aff
    """
    def __init__(self, test_csv_path: str, provided_stats: Tuple[float, float], verbose: bool = True):
        self.verbose = verbose
        test_df = pd.read_csv(test_csv_path)

        required_cols = ['binder_sequence', 'target_sequence', 'log_Aff']
        missing = [col for col in required_cols if col not in test_df.columns]
        if missing:
            raise KeyError(f"Test CSV missing required columns: {missing}")

        self.mean, self.std = provided_stats
        self.data = test_df.to_dict('records')
        self._log(f"\n[TestDataset] Using PROVIDED stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        self._log(f"[TestDataset] Loaded {len(self.data)} test samples")
        self._log(f"[TestDataset] Unique targets: {test_df['target_sequence'].nunique()}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        raw_aff = float(row["log_Aff"])
        # Normalize: lower predicted_affinity = better binder (natural log Kd scale)
        norm_aff = (raw_aff - self.mean) / (self.std + 1e-8)

        return {
            "binder_seq": str(row["binder_sequence"]),
            "target_seq": str(row["target_sequence"]),
            "log_Aff": norm_aff,
        }


class BinaryClassificationTestDataset(Dataset):
    """
    Test dataset for binary classification (binder vs non-binder).
    Expected format: target_id, binder_sequence, target_sequence, is_binder
    """
    def __init__(self, test_csv_path: str, verbose: bool = True):
        self.verbose = verbose
        test_df = pd.read_csv(test_csv_path)

        required_cols = ['target_id', 'binder_sequence', 'target_sequence', 'is_binder']
        missing = [col for col in required_cols if col not in test_df.columns]
        if missing:
            raise KeyError(f"Binary test CSV missing required columns: {missing}")

        self.data = test_df.to_dict('records')
        self._log(f"\n[BinaryTestDataset] Loaded {len(self.data)} samples")
        self._log(f"[BinaryTestDataset] Unique targets: {test_df['target_id'].nunique()}")
        self._log(f"[BinaryTestDataset] Binders: {test_df['is_binder'].sum()}, "
                  f"Non-binders: {(~test_df['is_binder'].astype(bool)).sum()}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "binder_seq": str(row["binder_sequence"]),
            "target_seq": str(row["target_sequence"]),
            "target_id": row["target_id"],
            "is_binder": int(row["is_binder"]),
        }
