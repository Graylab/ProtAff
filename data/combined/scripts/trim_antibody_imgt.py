"""
Trim antibody sequences in lookup_data.csv to variable region using IMGT numbering.

Uses AbNumber (ANARCI backend) to identify exact variable region boundaries via
IMGT numbering, rather than approximate length thresholds.

For each binder sequence:
  - Single-chain: attempt Chain(seq, scheme='imgt'); extract variable region via chain.seq
  - Multi-chain (contains ':'): split on ':', number each chain separately, rejoin with ':'
  - Non-antibody / numbering failure: keep sequence as-is

Requires: abnumber (pip install abnumber), which depends on ANARCI.

Usage:
    python data/combined/scripts/trim_antibody_imgt.py --data-dir data/combined
"""

import argparse
import os
import warnings
from collections import Counter

import pandas as pd
from abnumber import Chain
from tqdm import tqdm

warnings.filterwarnings("ignore")


def trim_chain_imgt(seq: str) -> tuple[str, str]:
    """
    Attempt IMGT numbering on a single chain sequence.

    Returns (result_seq, status) where status is one of:
      'trimmed'      — variable region extracted, sequence shortened
      'already_var'  — numbering succeeded, sequence unchanged
      'failed'       — numbering failed (non-antibody or ANARCI error)
    """
    try:
        chain = Chain(seq, scheme="imgt")
        var_seq = chain.seq
        if len(var_seq) < len(seq):
            return var_seq, "trimmed"
        return seq, "already_var"
    except Exception:
        return seq, "failed"


def trim_seq_imgt(seq: str) -> tuple[str, str]:
    """
    Trim a sequence to its variable region using IMGT numbering.

    Returns (trimmed_seq, action) where action is one of:
      'trimmed'           — variable region extracted successfully (sequence changed)
      'kept_already_var'  — numbering succeeded but sequence unchanged
      'kept_no_numbering' — numbering failed (non-antibody or ANARCI error)
    """
    chains = seq.split(":")

    if len(chains) == 1:
        result, status = trim_chain_imgt(seq)
        if status == "trimmed":
            return result, "trimmed"
        elif status == "already_var":
            return result, "kept_already_var"
        else:
            return result, "kept_no_numbering"

    # Multi-chain sequence
    trimmed_chains = []
    any_trimmed = False
    any_failed = False

    for chain_seq in chains:
        result, status = trim_chain_imgt(chain_seq)
        trimmed_chains.append(result)
        if status == "trimmed":
            any_trimmed = True
        elif status == "failed":
            any_failed = True

    joined = ":".join(trimmed_chains)

    if any_trimmed:
        return joined, "trimmed"
    elif any_failed:
        return joined, "kept_no_numbering"
    else:
        return joined, "kept_already_var"


def main():
    parser = argparse.ArgumentParser(
        description="Trim antibody sequences to variable region using IMGT numbering (AbNumber/ANARCI)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing lookup_data.csv; output written to <data-dir>/lookup_data_varonly_imgt.csv",
    )
    args = parser.parse_args()

    input_path = os.path.join(args.data_dir, "lookup_data.csv")
    output_path = os.path.join(args.data_dir, "lookup_data_varonly_imgt.csv")

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Total rows: {len(df):,}")

    binder_mask = df["type"] == "binder"
    print(f"Binder rows: {binder_mask.sum():,}")
    print(f"Non-binder rows: {(~binder_mask).sum():,}")
    print()

    actions = [""] * len(df)
    new_seqs = df["seq"].tolist()
    failed_indices = []

    binder_indices = df.index[binder_mask].tolist()
    for idx in tqdm(binder_indices, desc="IMGT numbering"):
        trimmed, action = trim_seq_imgt(df.at[idx, "seq"])
        new_seqs[idx] = trimmed
        actions[idx] = action
        if action == "kept_no_numbering":
            failed_indices.append(idx)

    df["seq"] = new_seqs
    df["_action"] = actions

    # Summary
    binder_actions = df.loc[binder_mask, "_action"]
    counts = Counter(binder_actions)
    total = binder_mask.sum()
    print()
    print("TRIMMING SUMMARY (binder rows)")
    print("-" * 50)
    for action, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {action:<35s} {cnt:>8,}  ({100 * cnt / total:.1f}%)")
    print()

    # Log sequences that failed numbering
    if failed_indices:
        print(f"Sequences that failed IMGT numbering ({len(failed_indices)}):")
        for idx in failed_indices[:20]:
            row = df.loc[idx]
            seq = row["seq"]
            preview = seq[:60] + "..." if len(seq) > 60 else seq
            print(f"  row {idx}: id={row.get('id', 'N/A')}, len={len(seq)}, seq={preview}")
        if len(failed_indices) > 20:
            print(f"  ... and {len(failed_indices) - 20} more")
        print()

    df.drop(columns=["_action"], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved trimmed data to {output_path}")


if __name__ == "__main__":
    main()
