import numpy as np
import pandas as pd
from typing import Optional, Tuple


def group_split(
    df: pd.DataFrame,
    col: str,
    ratio: float,
    seed: int,
    verbose: bool = True,
    label_col: Optional[str] = None,
    n_bins: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame by group column so that validation groups are completely
    unseen in training. Singletons (groups with only 1 sample) go to train.

    When ``label_col`` is provided the multi-sample groups are stratified by the
    per-group median of that column so that validation contains a representative
    mix of label values (e.g. high/low affinity targets).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    col : str
        Column to group by (e.g. 'target_id', 'filename', 'protein_a').
    ratio : float
        Fraction of *multi-sample groups* assigned to training.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print split statistics.
    label_col : str or None
        Column whose per-group median is used for stratified binning.
        When None the original random-shuffle behaviour is used.
    n_bins : int
        Number of quantile bins for stratification (default 4).

    Returns
    -------
    train_df, val_df : Tuple[pd.DataFrame, pd.DataFrame]
    """
    rng = np.random.default_rng(seed)

    counts = df[col].value_counts()
    singleton_groups = counts[counts == 1].index.tolist()
    multi_sample_groups = counts[counts > 1].index.tolist()

    if label_col is not None:
        val_groups_list, train_groups_multi_list = _stratified_split(
            df, col, multi_sample_groups, label_col, n_bins, ratio, rng, verbose,
        )
    else:
        rng.shuffle(multi_sample_groups)
        val_count = max(1, int(len(multi_sample_groups) * (1 - ratio)))
        val_groups_list = multi_sample_groups[:val_count]
        train_groups_multi_list = multi_sample_groups[val_count:]

    val_groups = set(val_groups_list)
    train_groups = set(train_groups_multi_list) | set(singleton_groups)

    train_df = df[df[col].isin(train_groups)].copy()
    val_df = df[df[col].isin(val_groups)].copy()

    # Verify no overlap
    overlap = train_groups & val_groups
    if len(overlap) > 0:
        raise ValueError(f"Train/Val {col} overlap detected: {overlap}")

    if verbose:
        print(f"[Split] Group column: '{col}'")
        print(f"[Split] Total groups: {len(counts)}")
        print(f"[Split]   Singleton (1 sample): {len(singleton_groups)}")
        print(f"[Split]   Multi-sample (>1):    {len(multi_sample_groups)}")
        if label_col is not None:
            print(f"[Split] Stratified by: '{label_col}' ({n_bins} bins)")
        print(f"[Split] Train groups: {len(train_groups)} | Val groups: {len(val_groups)} (UNSEEN)")
        print(f"[Split] Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")
        print(f"[Split] Overlap check: {len(overlap)} (MUST be 0)")

    return train_df, val_df


def _stratified_split(
    df: pd.DataFrame,
    col: str,
    multi_sample_groups: list,
    label_col: str,
    n_bins: int,
    ratio: float,
    rng: np.random.Generator,
    verbose: bool,
) -> Tuple[list, list]:
    """Split multi-sample groups into val/train lists, stratified by label."""

    # Per-group median of the label column
    group_medians = (
        df[df[col].isin(multi_sample_groups)]
        .groupby(col)[label_col]
        .median()
    )

    # Separate groups with NaN median — they go straight to train
    nan_mask = group_medians.isna()
    nan_groups = group_medians[nan_mask].index.tolist()
    valid_medians = group_medians[~nan_mask]

    if len(valid_medians) < 2:
        # Not enough groups to stratify — fall back to random shuffle
        if verbose:
            print(f"[Split] Stratification skipped: only {len(valid_medians)} groups with valid '{label_col}'")
        all_groups = list(valid_medians.index) + nan_groups
        rng.shuffle(all_groups)
        val_count = max(1, int(len(all_groups) * (1 - ratio)))
        return all_groups[:val_count], all_groups[val_count:]

    # Bin group medians into quantiles
    try:
        bins = pd.qcut(valid_medians, q=n_bins, duplicates="drop")
    except ValueError:
        # All identical values or similar edge case — fall back
        if verbose:
            print(f"[Split] Stratification failed (qcut): falling back to random shuffle")
        all_groups = list(valid_medians.index) + nan_groups
        rng.shuffle(all_groups)
        val_count = max(1, int(len(all_groups) * (1 - ratio)))
        return all_groups[:val_count], all_groups[val_count:]

    val_groups: list = []
    train_groups: list = []

    for bin_label, members in valid_medians.groupby(bins):
        grp_list = members.index.tolist()
        rng.shuffle(grp_list)

        if len(grp_list) == 1:
            # Strata with only 1 group → train (can't split)
            train_groups.extend(grp_list)
            if verbose:
                print(
                    f"[Split]   Bin {bin_label}: 1 group → all to train "
                    f"(range {members.min():.2f}–{members.max():.2f})"
                )
            continue

        val_count = max(1, int(len(grp_list) * (1 - ratio)))
        val_groups.extend(grp_list[:val_count])
        train_groups.extend(grp_list[val_count:])

        if verbose:
            print(
                f"[Split]   Bin {bin_label}: {len(grp_list)} groups → "
                f"{val_count} val / {len(grp_list) - val_count} train "
                f"(range {members.min():.2f}–{members.max():.2f})"
            )

    # NaN-median groups go to train
    train_groups.extend(nan_groups)
    if verbose and nan_groups:
        print(f"[Split]   NaN-median groups → train: {len(nan_groups)}")

    return val_groups, train_groups
