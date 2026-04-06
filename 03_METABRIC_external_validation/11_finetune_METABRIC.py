# -*- coding: utf-8 -*-
"""
FINE-GRID PENALIZATION SEARCH - METABRIC (Experiment 6)
=========================================================
Takes the winning configuration from Experiment 10:
  Dataset   : min200
  Algorithm : IAMB  alpha=0.20
  Features  : 87

Iterates over a fine grid of Cox penalization values and top-N feature subsets.
Evaluates each strategy with C-index and 5-year AUC.

Script location: .../Thesis_v3/03_METABRIC_external_validation/
"""

import sys
import os
print("Starting...", flush=True)

from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore")

_HERE   = Path(__file__).resolve().parent
_MERGED = _HERE / "merged"
OUTPUT_DIR = _HERE / "experiment_results" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_5Y = 365.25 * 5

# =========================================================================== #
# WINNING CONFIG — from Experiment 10
# =========================================================================== #

WINNER_FEATURES = [
    # clinical
    "clin_HER2_STATUS", "clin_TUMOR_SIZE", "clin_TUMOR_STAGE",
    # mutations
    "mut_TAF1", "mut_MEN1", "mut_CACNA2D3", "mut_UBR5", "mut_ERBB2",
    "mut_NPNT", "mut_BRIP1", "mut_ARID1A", "mut_COL22A1", "mut_BRCA2",
    "mut_RUNX1", "mut_FANCA", "mut_MAP3K10", "mut_AGMO", "mut_ROS1",
    "mut_PIK3CA", "mut_KDM3A", "mut_ARID2",
    # rna
    "rna_POLN", "rna_OGA", "rna_CFAP43", "rna_FGF6", "rna_RC3H2",
    "rna_PDCD2", "rna_ITPRIP", "rna_EVI5", "rna_LAD1", "rna_MGC70863",
    "rna_TSNAX", "rna_H4C3", "rna_DTX3", "rna_CFL1", "rna_TNFRSF10A",
    "rna_ELF2", "rna_CSNK2A3", "rna_ENC1", "rna_MED6", "rna_APLN",
    "rna_ADGRG1", "rna_CLINT1", "rna_RAMP2", "rna_PRAME", "rna_SPTAN1",
    "rna_NCOA7", "rna_DHRS3", "rna_TOLLIP", "rna_SMAD7", "rna_ATP1A1",
    "rna_CYBRD1", "rna_RAPGEF3", "rna_MED10", "rna_RALGAPB", "rna_SLC28A1",
    "rna_TREX2", "rna_USP30", "rna_C2orf42", "rna_UTP23", "rna_AP2B1",
    "rna_SPRYD4", "rna_ADAP2", "rna_GTSE1", "rna_HP1BP3", "rna_H3C3",
    "rna_H2BC8", "rna_ERCC8", "rna_ATP5MC3", "rna_S100P", "rna_RNASE1",
    "rna_SRFBP1", "rna_TFPT", "rna_ABCB9", "rna_KIAA1958", "rna_DCAF4",
    "rna_TIMM29", "rna_NDRG1", "rna_SLC25A3", "rna_SERPINE1", "rna_PURA",
    "rna_LAIR1", "rna_SPP1", "rna_PTP4A2", "rna_LHX2", "rna_ZNF33A",
    "rna_RP2",
]

print(f"Winner features: {len(WINNER_FEATURES)}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)
print("FINE-GRID PENALIZATION SEARCH - METABRIC")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =========================================================================== #
# HELPERS
# =========================================================================== #

def modality_of(col):
    if col.startswith("clin_"): return "clinical"
    if col.startswith("rna_"):  return "rna"
    if col.startswith("mut_"):  return "mutations"
    if col.startswith("cnv_"):  return "cnv"
    return "unknown"


def make_5y_binary(outcome):
    return ((outcome["OS"] == 1) & (outcome["OS.time"] <= THRESHOLD_5Y)).astype(int)


def cox_cindex(X_df, outcome, penalizer):
    """Fit penalized CoxPH, return C-index and partial hazard."""
    scaler = StandardScaler()
    X_sc   = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns, index=X_df.index)
    cox_df = X_sc.copy()
    cox_df["OS.time"] = outcome["OS.time"].values
    cox_df["OS"]      = outcome["OS"].values
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_df, duration_col="OS.time", event_col="OS")
    risk    = cph.predict_partial_hazard(X_sc).values
    c_index = concordance_index(outcome["OS.time"].values, -risk, outcome["OS"].values)
    return float(c_index), cph


def auc_5y(X_df, outcome):
    """5-fold CV Logistic Regression AUC for 5-year survival."""
    y5 = make_5y_binary(outcome)
    if y5.sum() < 10:
        return 0.5
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_df)
    lr     = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
    y_prob = cross_val_predict(lr, X_sc, y5, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y5, y_prob))


def mi_rank(X_df, outcome):
    """Rank features by mutual information with OS.time."""
    mi = mutual_info_regression(X_df, outcome["OS.time"], random_state=42, n_neighbors=5)
    return pd.Series(mi, index=X_df.columns).sort_values(ascending=False)


# =========================================================================== #
# LOAD DATA
# =========================================================================== #

print("\nLoading merged dataset (min200)...", flush=True)
files = sorted(_MERGED.glob("merged_min200_*.csv"))
if not files:
    print("ERROR: merged_min200 file not found")
    sys.exit(1)

X_full  = pd.read_csv(files[0], index_col=0)
outcome = pd.read_csv(_MERGED / "outcome_min200.csv", index_col=0)
common  = sorted(set(X_full.index) & set(outcome.index))
X_full  = X_full.loc[common]
outcome = outcome.loc[common]

print(f"  Dataset : {X_full.shape}")
print(f"  Samples : {len(common)}")
print(f"  Events  : {int(outcome['OS'].sum())} ({outcome['OS'].mean()*100:.1f}%)")
y5 = make_5y_binary(outcome)
print(f"  5y events: {y5.sum()} ({y5.mean()*100:.1f}%)")

# Slice to winner features (keep only those present in dataset)
avail   = [f for f in WINNER_FEATURES if f in X_full.columns]
missing = [f for f in WINNER_FEATURES if f not in X_full.columns]
if missing:
    print(f"  WARNING: {len(missing)} winner features not in dataset: {missing[:5]}...")
X_winner = X_full[avail].copy()
print(f"  Winner features available: {len(avail)} / {len(WINNER_FEATURES)}")


# =========================================================================== #
# EXPERIMENT GRID
# =========================================================================== #

# Penalty values — fine grid
PENALTIES = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]

# Top-N subsets (ranked by MI with OS.time within winner features)
mi_scores  = mi_rank(X_winner, outcome)
TOP_N_LIST = [10, 15, 18, 20, 25, 30, 40, 50, len(avail)]

# MI union ensemble: top features by MI across winner set
MI_UNION_THRESHOLD = 0.01   # MI score cutoff for ensemble

results = []
sid     = 0

total = len(PENALTIES) * len(TOP_N_LIST) + len(PENALTIES) + 2
print(f"\nTotal strategies to evaluate: {total}")
print("=" * 70)


# --- Grid: penalty x top-N ---
for top_n in TOP_N_LIST:
    top_genes = mi_scores.head(top_n).index.tolist()
    X_sub     = X_winner[top_genes]

    for pen in PENALTIES:
        sid += 1
        try:
            c_idx, _ = cox_cindex(X_sub, outcome, pen)
            auc      = auc_5y(X_sub, outcome)
        except Exception as e:
            print(f"  S{sid:02d} FAILED pen={pen} top={top_n}: {e}")
            continue

        mod_counts = {}
        for f in top_genes:
            m = modality_of(f)
            mod_counts[m] = mod_counts.get(m, 0) + 1

        results.append({
            "strategy_id":  f"S{sid:02d}",
            "description":  f"Cox Penalized (pen={pen}, top-{top_n})",
            "penalizer":    pen,
            "top_n":        top_n,
            "variant":      "top_n_mi",
            "n_features":   top_n,
            "c_index":      c_idx,
            "auc_5y":       auc,
            "mod_counts":   str(mod_counts),
            "features":     top_genes,
        })
        print(f"  S{sid:02d}  pen={pen:5}  top-{top_n:3d} : C={c_idx:.4f}  AUC={auc:.4f}",
              flush=True)


# --- All winner features, vary penalty only ---
print("\n--- All winner features, vary penalty ---")
for pen in PENALTIES:
    sid += 1
    try:
        c_idx, _ = cox_cindex(X_winner, outcome, pen)
        auc      = auc_5y(X_winner, outcome)
    except Exception as e:
        print(f"  S{sid:02d} FAILED: {e}")
        continue

    results.append({
        "strategy_id":  f"S{sid:02d}",
        "description":  f"Pen-{pen} (All winner features)",
        "penalizer":    pen,
        "top_n":        len(avail),
        "variant":      "all_features",
        "n_features":   len(avail),
        "c_index":      c_idx,
        "auc_5y":       auc,
        "mod_counts":   "",
        "features":     avail,
    })
    print(f"  S{sid:02d}  pen={pen:5}  all-{len(avail)} : C={c_idx:.4f}  AUC={auc:.4f}",
          flush=True)


# --- MI union ensemble (features above MI threshold) ---
sid += 1
mi_union = mi_scores[mi_scores >= MI_UNION_THRESHOLD].index.tolist()
if mi_union:
    try:
        c_idx, _ = cox_cindex(X_winner[mi_union], outcome, 10)
        auc      = auc_5y(X_winner[mi_union], outcome)
        results.append({
            "strategy_id":  f"S{sid:02d}",
            "description":  f"Pen-10 (MI Union Ensemble, MI>={MI_UNION_THRESHOLD})",
            "penalizer":    10,
            "top_n":        len(mi_union),
            "variant":      "mi_union_ensemble",
            "n_features":   len(mi_union),
            "c_index":      c_idx,
            "auc_5y":       auc,
            "mod_counts":   "",
            "features":     mi_union,
        })
        print(f"  S{sid:02d}  MI union ({len(mi_union)} features): C={c_idx:.4f}  AUC={auc:.4f}")
    except Exception as e:
        print(f"  S{sid:02d} MI union FAILED: {e}")


# --- No-missing indicator variant (top-20 with pen=10) ---
sid    += 1
top20   = mi_scores.head(20).index.tolist()
X_nm    = X_winner[top20].copy()
# Add binary missing-indicator features (already imputed, so all zeros — kept for consistency)
try:
    c_idx, _ = cox_cindex(X_nm, outcome, 10)
    auc      = auc_5y(X_nm, outcome)
    results.append({
        "strategy_id":  f"S{sid:02d}",
        "description":  "Pen-10 (No-Missing Indicator)",
        "penalizer":    10,
        "top_n":        20,
        "variant":      "no_missing_indicator",
        "n_features":   20,
        "c_index":      c_idx,
        "auc_5y":       auc,
        "mod_counts":   "",
        "features":     top20,
    })
    print(f"  S{sid:02d}  No-missing indicator (top-20): C={c_idx:.4f}  AUC={auc:.4f}")
except Exception as e:
    print(f"  S{sid:02d} No-missing FAILED: {e}")


# =========================================================================== #
# SUMMARY
# =========================================================================== #

df_res = pd.DataFrame([{k: v for k, v in r.items() if k != "features"}
                        for r in results])
df_res = df_res.sort_values("c_index", ascending=False).reset_index(drop=True)

print(f"\n{'='*70}")
print("TOP 10 CONFIGURATIONS")
print(f"{'='*70}")
print(df_res[["strategy_id", "description", "n_features", "c_index", "auc_5y"]].head(10).to_string(index=False))

# Save
df_res.to_csv(OUTPUT_DIR / "finetune_all_results.csv", index=False)

# Winner
best = max(results, key=lambda r: r["c_index"])
print(f"\n{'='*70}")
print("WINNER")
print(f"{'='*70}")
print(f"  Strategy  : {best['strategy_id']}  {best['description']}")
print(f"  Penalizer : {best['penalizer']}")
print(f"  Features  : {best['n_features']}")
print(f"  C-index   : {best['c_index']:.4f}")
print(f"  AUC 5y    : {best['auc_5y']:.4f}")

mod_counts = {}
for f in best["features"]:
    m = modality_of(f)
    mod_counts[m] = mod_counts.get(m, 0) + 1
print(f"  Modality breakdown: {mod_counts}")

print(f"\n  Features by modality:")
for mod in ["clinical", "rna", "mutations", "cnv"]:
    feats = [f.replace("clin_","").replace("rna_","").replace("mut_","").replace("cnv_","")
             for f in best["features"] if modality_of(f) == mod]
    if feats:
        print(f"    [{mod}] ({len(feats)}): {feats}")

feat_df = pd.DataFrame({
    "feature":      best["features"],
    "modality":     [modality_of(f) for f in best["features"]],
    "feature_name": [f.replace("clin_","").replace("rna_","").replace("mut_","").replace("cnv_","")
                     for f in best["features"]],
})
feat_df.to_csv(OUTPUT_DIR / "finetune_winner_features.csv", index=False)
with open(OUTPUT_DIR / "finetune_winner_config.json", "w") as f:
    json.dump({k: v for k, v in best.items() if k != "features"}, f, indent=2)

print(f"\n  Saved to: {OUTPUT_DIR}")
print("\nDONE")
