import sys
import os
import torch
import hydra
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Fix: Allow importing from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import build_model
from transformers import EsmTokenizer
from peft import PeftModel

# ----------------------------------------------------------------------
# 1. Inference Dataset & Collators
# ----------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
        def find_col(candidates):
            for c in candidates:
                if c in self.df.columns: return c
            return None

        self.b_col = find_col(["binder_sequence", "binder_seq", "binder", "seq_1", "heavy_chain", "cdr3"])
        self.t_col = find_col(["target_sequence", "target_seq", "target", "seq_2", "antigen"])
        
        if not self.b_col or not self.t_col:
            raise ValueError("CSV missing required binder/target columns.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "binder_seq": str(row[self.b_col]),
            "target_seq": str(row[self.t_col])
        }

class InferenceCollator:
    """
    Unified Inference Collator supporting Concat and Cross-Attention.
    """
    def __init__(self, tokenizer, arch="concat", max_length=1024):
        self.tokenizer = tokenizer
        self.arch = arch
        self.max_length = max_length

    def __call__(self, batch):
        eos = self.tokenizer.eos_token
        binder_seqs = [x["binder_seq"].replace(":", eos) for x in batch]
        target_seqs = [x["target_seq"].replace(":", eos) for x in batch]

        if self.arch in ["cross_attn", "interaction_map"]:
            # Decoupled tokenization for Late Fusion 
            b_enc = self.tokenizer(binder_seqs, padding=True, truncation=True, 
                                   max_length=self.max_length, return_tensors="pt")
            t_enc = self.tokenizer(target_seqs, padding=True, truncation=True, 
                                   max_length=self.max_length, return_tensors="pt")
            return {
                "binder_ids": b_enc["input_ids"],
                "binder_mask": b_enc["attention_mask"],
                "target_ids": t_enc["input_ids"],
                "target_mask": t_enc["attention_mask"]
            }
        else:
            # Traditional Concat logic
            b_encoded = self.tokenizer(binder_seqs, add_special_tokens=False)
            t_encoded = self.tokenizer(target_seqs, add_special_tokens=False)
            
            ids_list, mask_list = [], []
            cls_id, eos_id = self.tokenizer.cls_token_id, self.tokenizer.eos_token_id
            
            for b_ids, t_ids in zip(b_encoded["input_ids"], t_encoded["input_ids"]):
                budget = self.max_length - 3
                if len(b_ids) + len(t_ids) > budget:
                    t_ids = t_ids[:max(0, budget - len(b_ids))]
                    b_ids = b_ids[:budget]
                
                full_ids = [cls_id] + b_ids + [eos_id] + t_ids + [eos_id]
                ids_list.append(torch.tensor(full_ids, dtype=torch.long))
                mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

            return {
                "input_ids": pad_sequence(ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                "attention_mask": pad_sequence(mask_list, batch_first=True, padding_value=0)
            }

# ----------------------------------------------------------------------
# 2. Main Inference Routine
# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = cfg.model.get("arch", "concat")
    print(f"[System] Device: {device} | Architecture: {arch}")

    # Tagging and Path Logic (Maintained from your snippet)
    input_path = Path(cfg.input_csv)
    if cfg.get("tag"): exp_tag = cfg.tag
    else:
        path_obj = Path(cfg.model_path)
        if path_obj.is_file(): path_obj = path_obj.parent
        generic_names = ["saved_model", "checkpoints", "best_model", "last_model"]
        while path_obj.name in generic_names or path_obj.name.startswith("checkpoint-"):
            path_obj = path_obj.parent
        parent_name = path_obj.parent.name
        exp_tag = f"{parent_name}_{path_obj.name}" if (len(parent_name) == 10 and parent_name[0] == '2') else path_obj.name

    result_root = Path(cfg.get("base_results_dir", "outputs/inference")) / exp_tag
    dataset_dir = result_root / input_path.stem
    counter = 0
    while (dataset_dir / f"v{counter}").exists(): counter += 1
    save_dir = dataset_dir / f"v{counter}"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / cfg.get("output_filename", "predictions.csv")

    # Load Model
    base_model = build_model(cfg)
    model = PeftModel.from_pretrained(base_model, cfg.model_path)
    model.to(device).eval()
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)

    # Data Loading
    df = pd.read_csv(input_path)
    dataloader = DataLoader(
        InferenceDataset(df), 
        batch_size=cfg.get("batch_size", 16),
        collate_fn=InferenceCollator(tokenizer, arch=arch, max_length=cfg.model.get("max_length", 2048 if arch=="concat" else 1024)),
        num_workers=cfg.get("num_workers", 0)
    )

    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if arch in ["cross_attn", "interaction_map"]:
                outputs = model(
                    binder_ids=batch['binder_ids'], binder_mask=batch['binder_mask'],
                    target_ids=batch['target_ids'], target_mask=batch['target_mask']
                )
            else:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
            predictions.extend(outputs.reshape(-1).float().cpu().numpy().tolist())

    df["predicted_affinity"] = predictions
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Results saved to: {output_path}")

if __name__ == "__main__":
    main()