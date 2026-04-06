# -*- coding: utf-8 -*-
"""
MARKOV BLANKET - CNV (METABRIC)
=================================
V8 composite dataset, IAMB, alpha=0.05.
Runs twice: min_features=100 and min_features=200.
Padding source: FDR<0.05 genes ranked by composite score.

Script location: .../Thesis_v3/03_METABRIC_external_validation/
pyCausalFS    : .../Thesis_v3/pyCausalFS/
"""

import sys
import os
print("[1/4] Starting...", flush=True)

from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_CNV      = _HERE / "cnv"
_PYC_ROOT = _HERE.parent / "pyCausalFS"

_candidates = [
    _PYC_ROOT / "pyCausalFS" / "pyCausalFS",
    _PYC_ROOT / "pyCausalFS",
    _PYC_ROOT,
]
_PYCAUSAL = next((p for p in _candidates if (p / "CBD").exists()), None)
if _PYCAUSAL is None:
    print("ERROR: CBD package not found. Searched:", flush=True)
    for p in _candidates:
        print(f"  {p}  ({'ok' if p.exists() else 'missing'})", flush=True)
    sys.exit(1)
sys.path.insert(0, str(_PYCAUSAL))
print(f"[2/4] pyCausalFS: {_PYCAUSAL}", flush=True)

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings("ignore")

try:
    from CBD.MBs.IAMB import IAMB
    import CBD.MBs.common.fisher_z_test as fz
    print("[3/4] CBD imports OK", flush=True)
except ImportError as e:
    print(f"ERROR importing CBD: {e}", flush=True)
    sys.exit(1)


# =========================================================================== #
# FISHER-Z PATCH
# =========================================================================== #

def partial_corr_coef(data, x, y, z=None, ridge_lambda=1e-6):
    if z is None:
        has_z = False
    elif isinstance(z, (int, np.integer)):
        has_z = True
        z = [int(z)]
    elif hasattr(z, "__len__"):
        has_z = len(z) > 0
        if has_z:
            z = [int(zi) for zi in z]
    else:
        has_z = True
        z = [int(z)]

    if not has_z:
        var_x, var_y = data[x, x], data[y, y]
        if var_x < 1e-10 or var_y < 1e-10:
            return 0.0
        r = data[x, y] / np.sqrt(var_x * var_y)
        return float(np.clip(r, -0.999999, 0.999999))

    vars_list = [x, y] + z
    n = len(vars_list)
    sub_cov = np.array([[data[vi, vj] for vj in vars_list] for vi in vars_list])
    sub_cov += ridge_lambda * np.eye(n)
    try:
        precision = np.linalg.inv(sub_cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(sub_cov)
    p_xx, p_yy, p_xy = precision[0, 0], precision[1, 1], precision[0, 1]
    if p_xx < 1e-10 or p_yy < 1e-10:
        return 0.0
    r = -p_xy / np.sqrt(p_xx * p_yy)
    return float(np.clip(r, -0.999999, 0.999999))


fz.partial_corr_coef = partial_corr_coef


# =========================================================================== #
# CONFIG
# =========================================================================== #

FILTERED_DIR      = str(_CNV / "statistical_filtered")
STATS_CACHE       = str(_CNV / "cnv_statistics_cache.csv")
DATASET_PATTERN   = "cnv_8_composite"
ALGORITHM         = "IAMB"
ALPHA             = 0.05
MB_MIN_THRESHOLDS = [100, 200]

OUTPUT_DIRS = {
    100: str(_CNV / "mb_results_min100"),
    200: str(_CNV / "mb_results_min200"),
}
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

print("[4/4] Config OK", flush=True)
print("=" * 70)
print("MARKOV BLANKET - CNV (METABRIC)")
print("=" * 70)
print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset   : {DATASET_PATTERN}")
print(f"Algorithm : {ALGORITHM}  alpha={ALPHA}")
print(f"Thresholds: {MB_MIN_THRESHOLDS}")
print(f"Input dir : {FILTERED_DIR}")
print("=" * 70)


# =========================================================================== #
# HELPERS
# =========================================================================== #

def load_outcome():
    return pd.read_csv(os.path.join(FILTERED_DIR, "outcome.csv"), index_col=0)


def load_fdr_pool():
    stats_complete = os.path.join(FILTERED_DIR, "cnv_statistics_complete.csv")
    if os.path.exists(stats_complete):
        df = pd.read_csv(stats_complete)
        fdr_genes = df[df["fdr"] < 0.05].sort_values("composite")["gene"].tolist()
        if fdr_genes:
            print(f"  FDR<0.05 pool: {len(fdr_genes)} genes available for padding")
            return fdr_genes
        print("  WARNING: no FDR<0.05 genes - using full composite ranking for padding")
        return df.sort_values("composite")["gene"].tolist()
    if os.path.exists(STATS_CACHE):
        return pd.read_csv(STATS_CACHE)["gene"].tolist()
    print("  WARNING: no stats file found - padding disabled")
    return []


def pad_genes(selected, feature_names, min_n, fdr_pool):
    if len(selected) >= min_n:
        return selected
    have   = set(selected)
    extras = [g for g in fdr_pool if g in set(feature_names) and g not in have]
    padded = selected + extras[:min_n - len(selected)]
    print(f"  Padded (FDR<0.05): {len(selected)} raw -> {len(padded)} genes (min={min_n})")
    return padded


def evaluate_genes(cnv_df, outcome, gene_names):
    if not gene_names:
        return {"mean_mi": 0.0, "max_mi": 0.0, "c_index": 0.0}
    X  = cnv_df[gene_names]
    mi = mutual_info_regression(X, outcome["OS.time"], random_state=42, n_neighbors=5)
    try:
        c_idx = concordance_index(
            outcome["OS.time"].values,
            X.values.sum(axis=1),
            outcome["OS"].values,
        )
    except Exception:
        c_idx = 0.0
    return {"mean_mi": float(np.mean(mi)), "max_mi": float(np.max(mi)), "c_index": float(c_idx)}


def run_iamb(cnv_df, outcome, mb_min, fdr_pool, output_dir):
    feature_names = cnv_df.columns.tolist()

    X = cnv_df.values.astype(float) + np.random.normal(0.0, 1e-8, cnv_df.shape)
    y = outcome["OS.time"].values.astype(float)
    data_matrix = np.column_stack([X, y]).astype(float)
    target_idx  = X.shape[1]

    print(f"\n  Running IAMB (alpha={ALPHA}, {len(feature_names)} input genes)...")
    t0 = time.time()
    try:
        result = IAMB(data=data_matrix, target=target_idx,
                      is_discrete=False, alaph=ALPHA)
    except Exception as e:
        print(f"  ERROR in IAMB: {e}")
        return None
    elapsed = time.time() - t0

    mb_idx = list(result[0]) if isinstance(result, tuple) else (list(result) if result else [])
    mb_idx = [i for i in mb_idx if i != target_idx and 0 <= i < len(feature_names)]

    raw_genes    = [feature_names[i] for i in mb_idx]
    causal_genes = pad_genes(raw_genes, feature_names, mb_min, fdr_pool)
    metrics      = evaluate_genes(cnv_df, outcome, causal_genes)

    print(f"  Raw MB    : {len(raw_genes)} genes")
    print(f"  After pad : {len(causal_genes)} genes")
    print(f"  MI        : {metrics['mean_mi']:.4f}")
    print(f"  C-index   : {metrics['c_index']:.4f}")
    print(f"  Time      : {elapsed/60:.1f} min")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "IAMB_genes.txt"), "w") as f:
        f.write("\n".join(causal_genes))

    row = {
        "algorithm":        ALGORITHM,
        "alpha":            ALPHA,
        "mb_min_threshold": mb_min,
        "n_input_features": len(feature_names),
        "n_raw_mb_genes":   len(raw_genes),
        "n_causal_genes":   len(causal_genes),
        "mean_mi":          metrics["mean_mi"],
        "max_mi":           metrics["max_mi"],
        "c_index":          metrics["c_index"],
        "time_sec":         elapsed,
        "causal_genes":     causal_genes,
    }
    with open(os.path.join(output_dir, "IAMB_metrics.json"), "w") as f:
        json.dump({k: v for k, v in row.items() if k != "causal_genes"}, f, indent=2)

    return row


# =========================================================================== #
# MAIN
# =========================================================================== #

print("\nLoading data...", flush=True)
outcome  = load_outcome()
fdr_pool = load_fdr_pool()
print(f"  Outcome : {len(outcome)} samples, "
      f"{int(outcome['OS'].sum())} events ({outcome['OS'].mean()*100:.1f}%)")

all_files = sorted([
    f for f in os.listdir(FILTERED_DIR)
    if f.startswith(DATASET_PATTERN) and f.endswith("genes.csv")
])
if not all_files:
    print(f"ERROR: no file matching '{DATASET_PATTERN}' in {FILTERED_DIR}")
    sys.exit(1)
dataset_file = all_files[0]
print(f"\nDataset : {dataset_file}")

cnv_df = pd.read_csv(os.path.join(FILTERED_DIR, dataset_file), index_col=0)
print(f"Shape   : {cnv_df.shape[0]} samples x {cnv_df.shape[1]:,} genes")

common  = sorted(set(cnv_df.index) & set(outcome.index))
cnv_df  = cnv_df.loc[common]
outcome = outcome.loc[common]
print(f"Aligned : {len(common)} samples")

all_results = []
for mb_min in MB_MIN_THRESHOLDS:
    print(f"\n{'='*70}")
    print(f"RUNNING  mb_min = {mb_min}")
    print(f"{'='*70}")
    out_dir = os.path.join(OUTPUT_DIRS[mb_min], dataset_file.replace(".csv", ""))
    r = run_iamb(cnv_df, outcome, mb_min, fdr_pool, out_dir)
    if r:
        all_results.append(r)

print(f"\n{'='*70}")
print("COMPARISON  min100 vs min200")
print(f"{'='*70}")
for r in all_results:
    print(f"  min={r['mb_min_threshold']:3d}: "
          f"{r['n_raw_mb_genes']} raw -> {r['n_causal_genes']} genes | "
          f"MI={r['mean_mi']:.4f} | C-index={r['c_index']:.4f} | "
          f"{r['time_sec']/60:.1f} min")

if all_results:
    out_csv = str(_CNV / "mb_comparison_min100_vs_min200.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "causal_genes"}
                  for r in all_results]).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

print("\nDONE")
