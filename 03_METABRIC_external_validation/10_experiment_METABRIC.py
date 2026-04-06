# -*- coding: utf-8 -*-
"""
EXPERIMENT - METABRIC External Validation
==========================================
Runs MB feature selection on both merged datasets (min100, min200).
Evaluates each config with C-index and 5-year AUC.
Reports best config and selected features per modality.

Algorithms : IAMB, GSMB, MMMB (MMMB only at alpha=0.05)
Alphas     : 0.05, 0.10, 0.20
Datasets   : merged_min100, merged_min200

Script location: .../Thesis_v3/03_METABRIC_external_validation/
"""

import sys
import os
print("[1/4] Starting...", flush=True)

from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_MERGED   = _HERE / "merged"
_PYC_ROOT = _HERE.parent / "pyCausalFS"

_candidates = [
    _PYC_ROOT / "pyCausalFS" / "pyCausalFS",
    _PYC_ROOT / "pyCausalFS",
    _PYC_ROOT,
]
_PYCAUSAL = next((p for p in _candidates if (p / "CBD").exists()), None)
if _PYCAUSAL is None:
    print("ERROR: CBD package not found:", flush=True)
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
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings("ignore")

try:
    from CBD.MBs.IAMB import IAMB
    from CBD.MBs.GSMB import GSMB
    from CBD.MBs.MMMB.MMMB import MMMB
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

THRESHOLD_5Y = 365.25 * 5   # days

ALPHAS = [0.05, 0.10, 0.20]
ALGORITHMS_BY_ALPHA = {
    0.05: ["IAMB", "GSMB", "MMMB"],
    0.10: ["IAMB", "GSMB"],
    0.20: ["IAMB", "GSMB"],
}

OUTPUT_DIR = _HERE / "experiment_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[4/4] Config OK", flush=True)
print("=" * 70)
print("EXPERIMENT - METABRIC External Validation")
print("=" * 70)
print(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Algorithms : IAMB, GSMB, MMMB (MMMB only at alpha=0.05)")
print(f"Alphas     : {ALPHAS}")
print(f"Output     : {OUTPUT_DIR}")
print("=" * 70)


# =========================================================================== #
# HELPERS
# =========================================================================== #

def modality_of(col):
    """Return modality label from column prefix."""
    if col.startswith("clin_"):      return "clinical"
    if col.startswith("rna_"):       return "rna"
    if col.startswith("mut_"):       return "mutations"
    if col.startswith("cnv_"):       return "cnv"
    return "unknown"


def make_5y_binary(outcome):
    """Binary label: 1 = died within 5 years, 0 = survived/censored past 5y."""
    return ((outcome["OS"] == 1) & (outcome["OS.time"] <= THRESHOLD_5Y)).astype(int)


def evaluate(X_sel, outcome, feature_names):
    """Compute C-index (Cox) and 5-year AUC (Logistic CV)."""
    n_feat = len(feature_names)
    if n_feat == 0:
        return {"c_index": 0.5, "auc_5y": 0.5, "n_features": 0}

    # --- C-index via CoxPH ---
    try:
        scaler  = StandardScaler()
        X_sc    = pd.DataFrame(scaler.fit_transform(X_sel), columns=feature_names)
        cox_df  = X_sc.copy()
        cox_df["OS.time"] = outcome["OS.time"].values
        cox_df["OS"]      = outcome["OS"].values
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col="OS.time", event_col="OS")
        risk    = cph.predict_partial_hazard(X_sc).values
        c_index = concordance_index(outcome["OS.time"].values, -risk, outcome["OS"].values)
    except Exception:
        # Fallback: sum of features as risk score
        risk    = X_sel.values.sum(axis=1)
        c_index = concordance_index(outcome["OS.time"].values, risk, outcome["OS"].values)

    # --- 5-year AUC via cross-validated Logistic ---
    y_5y = make_5y_binary(outcome)
    try:
        if y_5y.sum() < 10:
            raise ValueError("Too few events for AUC")
        scaler2  = StandardScaler()
        X_sc2    = scaler2.fit_transform(X_sel)
        lr       = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
        y_prob   = cross_val_predict(lr, X_sc2, y_5y, cv=5, method="predict_proba")[:, 1]
        auc_5y   = roc_auc_score(y_5y, y_prob)
    except Exception:
        auc_5y = 0.5

    return {"c_index": float(c_index), "auc_5y": float(auc_5y), "n_features": n_feat}


def run_mb(name, data_matrix, target_idx, alpha):
    t0 = time.time()
    try:
        kwargs = dict(data=data_matrix, target=target_idx, is_discrete=False, alaph=alpha)
        if name == "IAMB":
            result = IAMB(**kwargs)
        elif name == "GSMB":
            result = GSMB(**kwargs)
        elif name == "MMMB":
            result = MMMB(**kwargs)
        elapsed = time.time() - t0
        mb_idx  = list(result[0]) if isinstance(result, tuple) else (list(result) if result else [])
        mb_idx  = [i for i in mb_idx if i != target_idx and 0 <= i < data_matrix.shape[1] - 1]
        return mb_idx, elapsed, None
    except Exception as e:
        return [], time.time() - t0, str(e)


def run_experiment(dataset_label, X, outcome):
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_label}  ({X.shape[0]} samples x {X.shape[1]} features)")
    print(f"{'='*70}")

    feature_names = X.columns.tolist()
    X_arr   = X.values.astype(float) + np.random.normal(0.0, 1e-8, X.shape)
    y       = outcome["OS.time"].values.astype(float)
    data_mx = np.column_stack([X_arr, y]).astype(float)
    target  = X_arr.shape[1]

    results = []

    for alpha in ALPHAS:
        for algo in ALGORITHMS_BY_ALPHA[alpha]:
            print(f"\n  {algo} alpha={alpha} ...", flush=True)
            mb_idx, elapsed, error = run_mb(algo, data_mx, target, alpha)

            if error:
                print(f"  ERROR: {error}")
                continue

            sel_names = [feature_names[i] for i in mb_idx]
            n_raw     = len(sel_names)

            if n_raw == 0:
                print(f"  No features selected — skipping evaluation")
                results.append({
                    "dataset": dataset_label, "algorithm": algo, "alpha": alpha,
                    "n_raw": 0, "c_index": 0.5, "auc_5y": 0.5, "time_min": elapsed/60,
                    "features": []
                })
                continue

            X_sel   = X[sel_names]
            metrics = evaluate(X_sel, outcome, sel_names)

            # Modality breakdown
            mod_counts = {}
            for f in sel_names:
                m = modality_of(f)
                mod_counts[m] = mod_counts.get(m, 0) + 1

            print(f"  Features : {n_raw}  {mod_counts}")
            print(f"  C-index  : {metrics['c_index']:.4f}")
            print(f"  AUC 5y   : {metrics['auc_5y']:.4f}")
            print(f"  Time     : {elapsed/60:.1f} min")

            results.append({
                "dataset":      dataset_label,
                "algorithm":    algo,
                "alpha":        alpha,
                "n_raw":        n_raw,
                "c_index":      metrics["c_index"],
                "auc_5y":       metrics["auc_5y"],
                "time_min":     elapsed / 60,
                "mod_counts":   mod_counts,
                "features":     sel_names,
            })

    return results


# =========================================================================== #
# MAIN
# =========================================================================== #

# Load datasets
datasets = {}
for threshold in [100, 200]:
    pattern = f"merged_min{threshold}_"
    files   = sorted(_MERGED.glob(f"{pattern}*.csv"))
    if not files:
        print(f"ERROR: no merged file for min={threshold} in {_MERGED}")
        sys.exit(1)
    label    = f"min{threshold}"
    X        = pd.read_csv(files[0], index_col=0)
    outcome  = pd.read_csv(_MERGED / f"outcome_min{threshold}.csv", index_col=0)
    common   = sorted(set(X.index) & set(outcome.index))
    datasets[label] = (X.loc[common], outcome.loc[common])
    print(f"Loaded {label}: {X.shape}  file: {files[0].name}")

print(f"\n5-year binary breakdown:")
for label, (X, outcome) in datasets.items():
    y5 = make_5y_binary(outcome)
    print(f"  {label}: died<5y={y5.sum()} ({y5.mean()*100:.1f}%)  "
          f"survived/censored={( y5==0).sum()}")

# Run experiments
all_results = []
for label, (X, outcome) in datasets.items():
    res = run_experiment(label, X, outcome)
    all_results.extend(res)

# =========================================================================== #
# SUMMARY
# =========================================================================== #

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ("features", "mod_counts")}
                        for r in all_results])
print(df_res.sort_values("c_index", ascending=False).to_string(index=False))

# Save full results
df_res.to_csv(OUTPUT_DIR / "all_results.csv", index=False)

# =========================================================================== #
# WINNER
# =========================================================================== #

print(f"\n{'='*70}")
print("BEST CONFIGURATION  (by C-index)")
print(f"{'='*70}")

best_row = max(all_results, key=lambda r: r["c_index"])
print(f"  Dataset   : {best_row['dataset']}")
print(f"  Algorithm : {best_row['algorithm']}  alpha={best_row['alpha']}")
print(f"  C-index   : {best_row['c_index']:.4f}")
print(f"  AUC 5y    : {best_row['auc_5y']:.4f}")
print(f"  Features  : {best_row['n_raw']}")
print(f"  Modality breakdown: {best_row['mod_counts']}")

# Feature list with modality labels
feat_df = pd.DataFrame({
    "feature":  best_row["features"],
    "modality": [modality_of(f) for f in best_row["features"]],
})
feat_df["feature_name"] = feat_df["feature"].str.replace(
    r"^(clin_|rna_|mut_|cnv_)", "", regex=True
)

print(f"\n  Selected features by modality:")
for mod, grp in feat_df.groupby("modality"):
    print(f"\n    [{mod}]  ({len(grp)} features)")
    for f in grp["feature_name"].tolist():
        print(f"      {f}")

# Save winner details
feat_df.to_csv(OUTPUT_DIR / "winner_features.csv", index=False)
with open(OUTPUT_DIR / "winner_config.json", "w") as f:
    json.dump({k: v for k, v in best_row.items() if k != "features"}, f, indent=2)

print(f"\n  Saved to: {OUTPUT_DIR}")
print("\nDONE")
