"""
FINAL ANALYSIS: Apply winning strategy (Cox penalizer>=5, top-20) to all 8 datasets.

Strategy: cox_pen5_top20 (C-index=0.8085 on 08_composite)
Dataset:  MERGE_continuous_outer/

Output:
  - final_results_all_datasets.csv   — metrics for all 8
  - final_selected_features/         — selected features per dataset
  - final_summary_report.txt         — human-readable summary
"""

import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    cwd = Path.cwd()
    SCRIPT_DIR = cwd if cwd.name == "MB" else cwd / "MB"

MERGE_DIR  = SCRIPT_DIR.parent / "MERGE_continuous_outer"
OUT_DIR    = SCRIPT_DIR / "results_final"
FEAT_DIR   = OUT_DIR / "selected_features"
OUT_DIR.mkdir(exist_ok=True)
FEAT_DIR.mkdir(exist_ok=True)

# Winning strategy parameters
PENALIZER   = 5.0
TOP_N       = 20
N_FOLDS     = 5
RANDOM_SEED = 42
SURVIVAL_YEARS = 5

DATASET_NAMES = {
    "01_ultra_conservative": "Ultra-Conservative",
    "02_conservative":       "Conservative",
    "03_standard":           "Standard",
    "04_fdr_significant":    "FDR-Significant",
    "05_balanced":           "Balanced",
    "06_correlation":        "Correlation",
    "07_top_correlated":     "Top-Correlated",
    "08_composite":          "Composite",
}

MODALITY_PREFIXES = ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]

print("=" * 70)
print("FINAL ANALYSIS: Cox pen=5 top-20 on all 8 datasets")
print("=" * 70)
print(f"Strategy:  Cox univariate, penalizer={PENALIZER}, top-{TOP_N} features")
print(f"CV:        {N_FOLDS}-fold cross-validated")
print(f"Input:     {MERGE_DIR.name}/")
print(f"Output:    {OUT_DIR}")

# ============================================================================
# HELPERS
# ============================================================================

def cox_topn(df, feat_cols, n, penalizer=5.0):
    sub = df[feat_cols + ["OS", "OS.time"]].dropna()
    pvals = {}
    for col in feat_cols:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(sub[[col,"OS","OS.time"]], duration_col="OS.time", event_col="OS")
            pvals[col] = cph.summary["p"].iloc[0]
        except Exception:
            pvals[col] = 1.0
    return sorted(pvals, key=lambda x: pvals[x])[:n]


def _cindex_manual(times, events, scores):
    n_conc = n_disc = n_tied = 0
    for i in range(len(times)):
        for j in range(i+1, len(times)):
            if events[i]==0 and events[j]==0: continue
            if times[i]==times[j]: continue
            e,l = (i,j) if times[i]<times[j] else (j,i)
            if not events[e]: continue
            if scores[e]>scores[l]: n_conc+=1
            elif scores[e]<scores[l]: n_disc+=1
            else: n_tied+=1
    total = n_conc+n_disc+n_tied
    return (n_conc+0.5*n_tied)/total if total>0 else 0.5


def cv_metrics(df, features, n_folds=N_FOLDS):
    if not features:
        return 0.5, 0.5
    sub = df[features+["OS","OS.time"]].dropna()
    if len(sub) < 40:
        return 0.5, 0.5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    threshold = SURVIVAL_YEARS * 365
    c_scores, auc_scores = [], []
    for tr, te in kf.split(sub):
        train, test = sub.iloc[tr], sub.iloc[te]
        try:
            cph = CoxPHFitter(penalizer=PENALIZER)
            cph.fit(train, duration_col="OS.time", event_col="OS")
            pred = cph.predict_partial_hazard(test).values
            c_scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, pred))
        except Exception:
            sc = StandardScaler().fit(train[features])
            score = sc.transform(test[features]).mean(1)
            c_scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, score))
        try:
            s5tr = train[(train["OS.time"]>=threshold)|(train["OS"]==1)]
            s5te = test[(test["OS.time"]>=threshold)|(test["OS"]==1)]
            ytr  = ((s5tr["OS"]==1)&(s5tr["OS.time"]<=threshold)).astype(int)
            yte  = ((s5te["OS"]==1)&(s5te["OS.time"]<=threshold)).astype(int)
            if ytr.sum()>=3 and (1-ytr).sum()>=3 and yte.sum()>=1:
                sc = StandardScaler()
                lr = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED)
                lr.fit(sc.fit_transform(s5tr[features]), ytr)
                auc_scores.append(roc_auc_score(
                    yte, lr.predict_proba(sc.transform(s5te[features]))[:,1]))
        except Exception:
            pass
    return (float(np.mean(c_scores)) if c_scores else 0.5,
            float(np.mean(auc_scores)) if auc_scores else 0.5)


def modality_breakdown(features):
    counts = {}
    for p in MODALITY_PREFIXES:
        n = sum(1 for f in features if f.startswith(p))
        if n: counts[p.rstrip("_")] = n
    return counts


# ============================================================================
# MAIN LOOP
# ============================================================================
print("\n" + "="*70)
print("PROCESSING ALL 8 DATASETS")
print("="*70)

results = []

for short_name, long_name in DATASET_NAMES.items():
    ds_file = MERGE_DIR / f"{short_name}.csv"
    if not ds_file.exists():
        print(f"\nSkipping {short_name} — file not found")
        continue

    print(f"\n{'─'*70}")
    print(f"[{short_name}]  {long_name}")

    df = pd.read_csv(ds_file, index_col=0)
    df = df.dropna(subset=["OS", "OS.time"])
    feat_cols = [c for c in df.columns if c not in ("OS", "OS.time")]

    n_samples = len(df)
    n_events  = int(df["OS"].sum())
    print(f"  Samples: {n_samples}  |  Events: {n_events}  |  Features: {len(feat_cols)}")

    # Feature selection
    t0 = time.time()
    selected = cox_topn(df, feat_cols, TOP_N, penalizer=PENALIZER)
    sel_time = time.time() - t0

    mod_counts = modality_breakdown(selected)
    print(f"  Selected: {len(selected)} features  ({sel_time:.1f}s)")
    print(f"  Modalities: {mod_counts}")

    # Evaluate
    c_idx, auc5 = cv_metrics(df, selected)
    print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}")

    # Save features
    feat_df = pd.DataFrame({
        "feature":  selected,
        "modality": [next((p.rstrip("_") for p in MODALITY_PREFIXES
                          if f.startswith(p)), "OTHER") for f in selected],
    })
    feat_df.to_csv(FEAT_DIR / f"{short_name}_features.csv", index=False)

    results.append({
        "dataset":       short_name,
        "dataset_name":  long_name,
        "n_samples":     n_samples,
        "n_events":      n_events,
        "n_features":    len(selected),
        "c_index":       round(c_idx, 4),
        "auc_5yr":       round(auc5, 4),
        **{f"n_{p.rstrip('_').lower()}": mod_counts.get(p.rstrip("_"), 0)
           for p in MODALITY_PREFIXES},
    })

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*70)
print("FINAL RESULTS — ALL 8 DATASETS")
print("="*70)

res_df = pd.DataFrame(results).sort_values("c_index", ascending=False).reset_index(drop=True)
display_cols = ["dataset_name","n_samples","n_events","n_features",
                "c_index","auc_5yr",
                "n_clin","n_rna","n_cnv","n_mut","n_prot","n_meth","n_mirna"]
print(res_df[display_cols].to_string(index=False))

res_df.to_csv(OUT_DIR / "final_results_all_datasets.csv", index=False)

# ============================================================================
# TEXT REPORT
# ============================================================================
best = res_df.iloc[0]

report_lines = [
    "FINAL ANALYSIS REPORT",
    "=" * 70,
    f"Strategy:  Cox univariate, penalizer={PENALIZER}, top-{TOP_N} features",
    f"Datasets:  MERGE_continuous_outer/ (outer join, continuous RNA, missingness indicators)",
    f"Eval:      {N_FOLDS}-fold cross-validated C-index + AUC-5yr",
    "",
    "RESULTS SUMMARY:",
    "-" * 70,
]
for _, row in res_df.iterrows():
    flag = " <- BEST" if row["dataset"] == best["dataset"] else ""
    report_lines.append(
        f"  {row['dataset_name']:22s}  C-index={row['c_index']:.4f}  "
        f"AUC-5yr={row['auc_5yr']:.4f}  n={row['n_samples']}{flag}"
    )

report_lines += [
    "",
    "BEST CONFIGURATION:",
    "-" * 70,
    f"  Dataset:   {best['dataset_name']}",
    f"  C-index:   {best['c_index']}",
    f"  AUC-5yr:   {best['auc_5yr']}",
    f"  Samples:   {best['n_samples']}  (events: {best['n_events']})",
    f"  Features:  {best['n_features']}",
    "",
    "  Modality breakdown:",
]
for p in MODALITY_PREFIXES:
    key = f"n_{p.rstrip('_').lower()}"
    n = best.get(key, 0)
    if n > 0:
        report_lines.append(f"    {p.rstrip('_'):8s}: {int(n)}")

report_lines += [
    "",
    "SELECTED FEATURES (best dataset):",
    "-" * 70,
]
best_feat_file = FEAT_DIR / f"{best['dataset']}_features.csv"
if best_feat_file.exists():
    best_feats = pd.read_csv(best_feat_file)
    for mod in best_feats["modality"].unique():
        report_lines.append(f"\n  {mod}:")
        for f in best_feats[best_feats["modality"]==mod]["feature"].tolist():
            report_lines.append(f"    {f}")

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(OUT_DIR / "final_summary_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n{'='*70}")
print(f"Saved to: {OUT_DIR}")
print(f"  final_results_all_datasets.csv")
print(f"  final_summary_report.txt")
print(f"  selected_features/  ({len(results)} files)")
