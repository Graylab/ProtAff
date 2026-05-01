#!/usr/bin/env python3
"""Prepare all-vs-all VHH × antigen pairs for the ab_ag_champloo benchmark.

Reads the champloo supplementary table, filters to the 91 non-excluded PDB systems
(matching the iPTM matrices), and generates all 91×91 = 8,281 pairs.

Usage:
    python data/pipelines/prepare_champloo_pairs.py \
        --output data/sources/ab_ag_champloo/champloo_allvsall.csv
"""
import argparse
import pandas as pd
from pathlib import Path

PROTAFF_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SUPP_TABLE = (
    PROTAFF_ROOT
    / "data/sources/ab_ag_champloo"
    / "Supplementary_Table_1_final_experimental_vhh_ag_systems.csv"
)
DEFAULT_MATRIX = (
    PROTAFF_ROOT
    / "data/sources/ab_ag_champloo/iptm_confidence_scores/af3_matrix_clean.csv"
)


def main():
    parser = argparse.ArgumentParser(description="Prepare champloo all-vs-all pairs")
    parser.add_argument(
        "--supp_table", default=str(DEFAULT_SUPP_TABLE),
        help="Path to the champloo supplementary table CSV",
    )
    parser.add_argument(
        "--matrix", default=str(DEFAULT_MATRIX),
        help="Path to an iPTM matrix CSV (used to determine the 91 PDB IDs)",
    )
    parser.add_argument(
        "--output", default=str(
            PROTAFF_ROOT / "data/sources/ab_ag_champloo/champloo_allvsall.csv"
        ),
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Load supplementary table
    supp = pd.read_csv(args.supp_table)
    print(f"Loaded {len(supp)} entries from supplementary table")

    # Get the 91 PDB IDs from the matrix
    matrix = pd.read_csv(args.matrix, nrows=0)
    matrix_pdb_ids = list(matrix.columns[1:])  # skip index column
    print(f"Found {len(matrix_pdb_ids)} PDB IDs in matrix")

    # Filter to non-excluded entries matching matrix PDB IDs
    non_excl = supp[supp["is_excluded"] != True].copy()
    non_excl = non_excl[non_excl["pdb_id"].isin(matrix_pdb_ids)]

    # Take first entry per PDB ID (some PDBs have multiple chain combos)
    representatives = non_excl.drop_duplicates(subset="pdb_id", keep="first")
    representatives = representatives.set_index("pdb_id")

    # Verify we have all 91
    missing = set(matrix_pdb_ids) - set(representatives.index)
    if missing:
        raise ValueError(f"Missing PDB IDs: {missing}")
    print(f"Using {len(representatives)} representative systems")

    # Generate all-vs-all pairs (VHH from row PDB × antigen from col PDB)
    rows = []
    for vhh_pdb in matrix_pdb_ids:
        vhh_seq = representatives.loc[vhh_pdb, "vhh_sequence"]
        for ag_pdb in matrix_pdb_ids:
            ag_seq = representatives.loc[ag_pdb, "antigen_sequence"]
            label = 1 if vhh_pdb == ag_pdb else 0
            rows.append({
                "pdb_vhh": vhh_pdb,
                "pdb_antigen": ag_pdb,
                "binder_sequence": vhh_seq,
                "target_sequence": ag_seq,
                "label": label,
            })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} pairs: {df['label'].sum()} positives, "
          f"{(df['label'] == 0).sum()} negatives")

    # Sanity checks
    n = len(matrix_pdb_ids)
    assert len(df) == n * n, f"Expected {n*n} pairs, got {len(df)}"
    assert df["label"].sum() == n, f"Expected {n} positives, got {df['label'].sum()}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
