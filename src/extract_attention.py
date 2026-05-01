import sys
import os
import torch
import hydra
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.inference import InferenceDataset, load_model, resolve_output_path
from src.datasets.collators import select_collator
from transformers import EsmTokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Device: {device}")

    input_path = Path(cfg.input_csv)
    model_path = Path(cfg.model_path)

    # Determine output directory
    if cfg.get("output_dir"):
        output_dir = Path(cfg.output_dir)
    else:
        # Reuse resolve_output_path logic, take its parent dir
        output_csv = resolve_output_path(cfg, input_path, model_path)
        output_dir = output_csv.parent / "attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg, device)
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)

    df = pd.read_csv(input_path)

    # --- Target perturbation experiments ---
    experiment = cfg.get("experiment", None)
    if experiment is not None:
        t_col = None
        for c in InferenceDataset.TARGET_CANDIDATES:
            if c in df.columns:
                t_col = c
                break
        if t_col is None:
            raise ValueError("Cannot find target column for perturbation")

        if experiment == "scramble":
            import random
            seed = cfg.get("scramble_seed", 42)
            rng = random.Random(seed)

            def scramble(seq):
                residues = list(seq)
                rng.shuffle(residues)
                return "".join(residues)

            df[t_col] = df[t_col].apply(scramble)
            print(f"[Experiment] Scrambled target sequences (seed={seed})")

        elif experiment == "constant":
            DEFAULT_GFP = (
                "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
                "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN"
                "RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY"
                "QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
            )
            constant_seq = cfg.get("constant_target_seq", None) or DEFAULT_GFP
            df[t_col] = constant_seq
            print(f"[Experiment] Replaced all targets with constant sequence (len={len(constant_seq)})")

        else:
            raise ValueError(f"Unknown experiment mode: {experiment}")

    dataset = InferenceDataset(df)
    max_length = cfg.model.get("max_length", 1024)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 16),
        collate_fn=select_collator(tokenizer, max_length, mode="inference"),
        num_workers=cfg.get("num_workers", 0),
    )

    # Collect results
    all_predictions = []
    all_binder_seqs = []
    all_target_seqs = []
    all_binder_lengths = []
    all_target_lengths = []
    # cross_attn_maps[layer_idx] = list of per-sample arrays
    cross_attn_maps = None
    all_pool_weights = []

    sample_idx = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(dataloader, desc="Extracting attention")):
            batch_device = {k: v.to(device) for k, v in batch.items()}

            scores, attn_weights_list, pool_weights = model(
                binder_ids=batch_device["binder_ids"],
                binder_mask=batch_device["binder_mask"],
                target_ids=batch_device["target_ids"],
                target_mask=batch_device["target_mask"],
                return_attn=True,
            )

            preds = scores.reshape(-1).float().cpu().numpy()
            all_predictions.extend(preds.tolist())

            binder_mask = batch["binder_mask"].cpu()
            target_mask = batch["target_mask"].cpu()
            binder_lens = binder_mask.sum(dim=1).numpy()
            target_lens = target_mask.sum(dim=1).numpy()

            pool_w = pool_weights.cpu().numpy()  # (B, binder_len_padded)
            n_layers = len(attn_weights_list)
            if cross_attn_maps is None:
                cross_attn_maps = [[] for _ in range(n_layers)]

            batch_size = len(preds)
            for i in range(batch_size):
                bl = int(binder_lens[i])
                tl = int(target_lens[i])
                # Store residue-only lengths (strip BOS/EOS special tokens)
                all_binder_lengths.append(bl - 2)
                all_target_lengths.append(tl - 2)

                # Pool weights: strip padding, then strip BOS/EOS
                all_pool_weights.append(pool_w[i, 1:bl - 1])

                # Cross-attention: strip padding, then strip BOS/EOS on both dims
                for layer_idx in range(n_layers):
                    # attn_weights_list[layer_idx] shape: (B, n_heads, binder_len, target_len)
                    aw = attn_weights_list[layer_idx][i].cpu().numpy()
                    cross_attn_maps[layer_idx].append(aw[:, 1:bl - 1, 1:tl - 1])

            # Get sequences from dataset
            start = batch_i * dataloader.batch_size
            for j in range(batch_size):
                item = dataset[start + j]
                all_binder_seqs.append(item["binder_seq"])
                all_target_seqs.append(item["target_seq"])

            sample_idx += batch_size

    # Build npz dict
    npz_data = {}
    n_samples = len(all_predictions)
    n_layers = len(cross_attn_maps)

    for layer_idx in range(n_layers):
        for s in range(n_samples):
            npz_data[f"cross_attn_L{layer_idx}_S{s}"] = cross_attn_maps[layer_idx][s]

    for s in range(n_samples):
        npz_data[f"pool_S{s}"] = all_pool_weights[s]

    npz_data["predictions"] = np.array(all_predictions, dtype=np.float32)
    npz_data["binder_seqs"] = np.array(all_binder_seqs, dtype=object)
    npz_data["target_seqs"] = np.array(all_target_seqs, dtype=object)
    npz_data["binder_lengths"] = np.array(all_binder_lengths, dtype=np.int32)
    npz_data["target_lengths"] = np.array(all_target_lengths, dtype=np.int32)

    npz_path = output_dir / "attention_maps.npz"
    np.savez(npz_path, **npz_data)
    print(f"[SUCCESS] Attention maps saved to: {npz_path}")
    print(f"  {n_samples} samples, {n_layers} cross-attn layers")

    # Summary CSV
    summary_df = pd.DataFrame({
        "binder_seq": all_binder_seqs,
        "target_seq": all_target_seqs,
        "binder_length": all_binder_lengths,
        "target_length": all_target_lengths,
        "predicted_affinity": all_predictions,
    })
    csv_path = output_dir / "summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
