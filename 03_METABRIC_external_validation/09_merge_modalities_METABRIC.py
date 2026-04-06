# -*- coding: utf-8 -*-
"""
MERGE MODALITIES - METABRIC
=============================
Merges 4 modalities into 2 final datasets:

  Dataset 1 (min100): clinical (all) + RNA MB min100 + mutations MB min100 + CNV MB min100
  Dataset 2 (min200): clinical (all) + RNA MB min200 + mutations MB min200 + CNV MB min200

Script location: .../Thesis_v3/03_METABRIC_external_validation/
"""

import sys
import os
print("Starting...", flush=True)

from pathlib import Path
import pandas as pd
import numpy as np

_HERE = Path(__file__).resolve().parent

# =========================================================================== #
# PATHS
# =========================================================================== #

# Clinical — preprocessed, all features, no MB
CLIN_FILE = _HERE / "clinical" / "statistical_filtered" / "clin_8_composite_Nfeatures.csv"

# MB gene lists per modality and threshold
MB_GENES = {
    "rna": {
        100: _HERE / "rna" / "mb_results_min100",
        200: _HERE / "rna" / "mb_results_min200",
    },
    "mutations": {
        100: _HERE / "mutations" / "mb_results_min100",
        200: _HERE / "mutations" / "mb_results_min200",
    },
    "cnv": {
        100: _HERE / "cnv" / "mb_results_min100",
        200: _HERE / "cnv" / "mb_results_min200",
    },
}

# Full preprocessed data per modality (to slice selected genes from)
MODALITY_DATA = {
    "rna":       _HERE / "rna"       / "statistical_filtered",
    "mutations": _HERE / "mutations" / "statistical_filtered",
    "cnv":       _HERE / "cnv"       / "statistical_filtered",
}

OUTCOME_FILE = _HERE / "rna" / "statistical_filtered" / "outcome.csv"
OUTPUT_DIR   = _HERE / "merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output : {OUTPUT_DIR}")


# =========================================================================== #
# HELPERS
# =========================================================================== #

def find_dataset_file(folder, pattern):
    """Find the single CSV matching pattern in folder."""
    folder = Path(folder)
    matches = sorted(folder.glob(f"{pattern}*genes.csv"))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {folder}")
    return matches[0]


def load_genes_from_mb(mb_dir, modality):
    """Load IAMB_genes.txt from the MB results folder for a given modality."""
    mb_dir = Path(mb_dir)
    # Find the subdirectory (named after the dataset file)
    subdirs = [d for d in mb_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectory found in {mb_dir}")
    genes_file = subdirs[0] / "IAMB_genes.txt"
    if not genes_file.exists():
        raise FileNotFoundError(f"IAMB_genes.txt not found in {subdirs[0]}")
    genes = genes_file.read_text().strip().splitlines()
    genes = [g.strip() for g in genes if g.strip()]
    print(f"  {modality:12s}: {len(genes)} genes loaded from {genes_file}")
    return genes


def load_modality_data(modality, genes):
    """Load full dataset CSV and slice to selected genes."""
    patterns = {"rna": "rna_8_composite", "mutations": "mut_8_composite", "cnv": "cnv_8_composite"}
    data_file = find_dataset_file(MODALITY_DATA[modality], patterns[modality])
    df = pd.read_csv(data_file, index_col=0)
    missing = [g for g in genes if g not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} genes not found in {modality} data — skipping")
    genes_ok = [g for g in genes if g in df.columns]
    return df[genes_ok]


def check_clinical(df):
    """Validate clinical data: no NaN, no Inf, no outcome leakage."""
    outcome_keywords = ["OS_MONTHS", "OS_STATUS", "VITAL", "SURVIVAL", "OS.TIME", "OS"]
    leaked = [c for c in df.columns if any(kw in c.upper() for kw in outcome_keywords)]
    if leaked:
        print(f"  WARNING: dropping outcome-related columns: {leaked}")
        df = df.drop(columns=leaked)
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=np.number).values).sum()
    print(f"  Clinical : {df.shape[1]} features | NaN={nan_count} | Inf={inf_count}")
    assert nan_count == 0, f"NaN values in clinical data: {nan_count}"
    assert inf_count == 0, f"Inf values in clinical data: {inf_count}"
    return df


def merge_and_save(threshold, outcome):
    print(f"\n{'='*70}")
    print(f"Building dataset  min={threshold}")
    print(f"{'='*70}")

    # Load MB gene lists
    genes = {}
    for mod in ["rna", "mutations", "cnv"]:
        genes[mod] = load_genes_from_mb(MB_GENES[mod][threshold], mod)

    # Load modality data sliced to selected genes
    dfs = {}
    for mod in ["rna", "mutations", "cnv"]:
        dfs[mod] = load_modality_data(mod, genes[mod])

    # Load and validate clinical
    # Auto-detect clinical file (N in filename varies)
    clin_dir = _HERE / "clinical" / "statistical_filtered"
    clin_files = sorted(clin_dir.glob("clin_8_composite_*features.csv"))
    if not clin_files:
        raise FileNotFoundError(f"No clinical composite file found in {clin_dir}")
    clin = pd.read_csv(clin_files[0], index_col=0)
    clin = check_clinical(clin)

    # Align all to common samples
    common = set(outcome.index)
    for name, df in [("clinical", clin)] + list(dfs.items()):
        common &= set(df.index)
    common = sorted(common)
    print(f"\n  Common samples across all modalities: {len(common)}")

    clin = clin.loc[common]
    for mod in dfs:
        dfs[mod] = dfs[mod].loc[common]
    outcome_aligned = outcome.loc[common]

    # Add modality prefix to avoid column name collisions
    clin    = clin.add_prefix("clin_")
    dfs["rna"]       = dfs["rna"].add_prefix("rna_")
    dfs["mutations"] = dfs["mutations"].add_prefix("mut_")
    dfs["cnv"]       = dfs["cnv"].add_prefix("cnv_")

    # Merge
    merged = pd.concat([clin, dfs["rna"], dfs["mutations"], dfs["cnv"]], axis=1)

    # Final checks
    nan_total = merged.isna().sum().sum()
    inf_total = np.isinf(merged.select_dtypes(include=np.number).values).sum()
    dup_cols  = merged.columns.duplicated().sum()

    print(f"\n  Merged shape     : {merged.shape}")
    print(f"  NaN              : {nan_total}")
    print(f"  Inf              : {inf_total}")
    print(f"  Duplicate cols   : {dup_cols}")
    print(f"\n  Column breakdown :")
    print(f"    clinical   : {clin.shape[1]}")
    print(f"    rna        : {dfs['rna'].shape[1]}")
    print(f"    mutations  : {dfs['mutations'].shape[1]}")
    print(f"    cnv        : {dfs['cnv'].shape[1]}")
    print(f"    TOTAL      : {merged.shape[1]}")

    assert nan_total == 0,  "NaN values in merged dataset!"
    assert inf_total == 0,  "Inf values in merged dataset!"
    assert dup_cols  == 0,  "Duplicate column names in merged dataset!"

    # Save
    fname   = f"merged_min{threshold}_{merged.shape[1]}features_{len(common)}samples.csv"
    out_path = OUTPUT_DIR / fname
    merged.to_csv(out_path)
    outcome_aligned.to_csv(OUTPUT_DIR / f"outcome_min{threshold}.csv")
    print(f"\n  Saved: {out_path}")

    return merged, outcome_aligned


# =========================================================================== #
# MAIN
# =========================================================================== #

print("\nLoading outcome...", flush=True)
outcome = pd.read_csv(OUTCOME_FILE, index_col=0)
print(f"  {len(outcome)} samples, {int(outcome['OS'].sum())} events ({outcome['OS'].mean()*100:.1f}%)")

results = {}
for threshold in [100, 200]:
    merged, out = merge_and_save(threshold, outcome)
    results[threshold] = merged

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for t, df in results.items():
    print(f"  min={t:3d}: {df.shape[0]} samples x {df.shape[1]} features")

print(f"\n  Files saved to: {OUTPUT_DIR}")
print("\nDONE")
