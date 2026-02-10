import sys
import os
import torch
import hydra
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

# Fix: Allow importing from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import build_model
from src.datasets.collators import CROSS_ATTN_ARCHS, select_collator
from transformers import EsmTokenizer
from peft import PeftModel


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class InferenceDataset(Dataset):
    """Dataset for inference with flexible column name detection."""

    BINDER_CANDIDATES = ["binder_sequence", "binder_seq", "binder", "seq_1", "heavy_chain", "cdr3"]
    TARGET_CANDIDATES = ["target_sequence", "target_seq", "target", "seq_2", "antigen"]

    def __init__(self, df):
        self.df = df
        self.b_col = self._find_col(self.BINDER_CANDIDATES)
        self.t_col = self._find_col(self.TARGET_CANDIDATES)
        if not self.b_col or not self.t_col:
            raise ValueError("CSV missing required binder/target columns.")

    def _find_col(self, candidates):
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {"binder_seq": str(row[self.b_col]), "target_seq": str(row[self.t_col])}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def resolve_output_path(cfg, input_path, model_path):
    """Build versioned output directory from model path and input CSV name."""
    path_obj = model_path if model_path.is_dir() else model_path.parent
    generic_names = {"saved_model", "checkpoints", "best_model", "last_model"}
    while path_obj.name in generic_names or path_obj.name.startswith("checkpoint-"):
        path_obj = path_obj.parent
    exp_tag = path_obj.name

    parts = model_path.parts
    idx = parts.index("outputs")
    model_rel_path = Path(*parts[idx + 1 : idx + 3])

    result_root = Path(cfg.get("base_results_dir", "outputs/inference"))
    dataset_dir = result_root / model_rel_path / exp_tag / input_path.stem

    counter = 0
    while (dataset_dir / f"v{counter}").exists():
        counter += 1
    save_dir = dataset_dir / f"v{counter}"
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir / cfg.get("output_filename", "predictions.csv")


def load_model(cfg, device):
    """Load base model with LoRA adapter."""
    base_model = build_model(cfg)
    model = PeftModel.from_pretrained(base_model, cfg.model_path)
    return model.to(device).eval()


def run_inference(model, dataloader, arch, device, binding_head="affinity"):
    """Run model inference and return predictions."""
    is_cross_attn = arch in CROSS_ATTN_ARCHS
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if is_cross_attn:
                kwargs = dict(
                    binder_ids=batch["binder_ids"], binder_mask=batch["binder_mask"],
                    target_ids=batch["target_ids"], target_mask=batch["target_mask"],
                )
                if arch == "binding":
                    kwargs["task"] = binding_head
                outputs = model(**kwargs)
            else:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            predictions.extend(outputs.reshape(-1).float().cpu().numpy().tolist())

    return predictions


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = cfg.model.get("arch", "concat")
    print(f"[System] Device: {device} | Architecture: {arch}")

    input_path = Path(cfg.input_csv)
    model_path = Path(cfg.model_path)
    output_path = resolve_output_path(cfg, input_path, model_path)

    model = load_model(cfg, device)
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)

    df = pd.read_csv(input_path)
    max_length = cfg.model.get("max_length", 2048 if arch == "concat" else 1024)
    dataloader = DataLoader(
        InferenceDataset(df),
        batch_size=cfg.get("batch_size", 16),
        collate_fn=select_collator(arch, tokenizer, max_length, mode="inference"),
        num_workers=cfg.get("num_workers", 0),
    )

    binding_head = cfg.model.get("binding_head", "affinity")
    predictions = run_inference(model, dataloader, arch, device, binding_head=binding_head)
    df["predicted_affinity"] = predictions
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
