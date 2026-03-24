"""
EXPERIMENT: Compare different feature selection fallback strategies
Dataset: 08_composite | IAMB α=0.05 (best so far)

Strategies tested:
  0. No fallback    — only truly significant features (may be very few)
  1. Cox univariate — top-N by univariate Cox p-value
  2. MI top-N       — top-N by mutual information (current fallback, baseline)
  3. MI + variance  — top-N by MI weighted by variance
  4. Elastic Net Cox — Cox-EN from full feature set
  5. MI permutation strict — significant only (alpha=0.01, no padding)

Metrics: 5-fold CV C-index + AUC-5yr
"""

import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    cwd = Path.cwd()
    SCRIPT_DIR = cwd if cwd.name == "MB" else cwd / "MB"

MERGE_DIR  = SCRIPT_DIR.parent / "MERGE_continuous_outer"
MB_DIR     = SCRIPT_DIR.parent / "RNA" / "mb_results"
OUT_DIR    = SCRIPT_DIR / "results_fallback_experiment"
OUT_DIR.mkdir(exist_ok=True)

DATASET    = "08_composite"
RNA_DS     = "rna_8_composite_1000genes"
ALGORITHM  = "IAMB"
ALPHA      = 0.05
N_FOLDS    = 5
RANDOM_SEED = 42
SURVIVAL_YEARS = 5
TOP_N      = 20   # target number of features for strategies that pick top-N

print(f"Dataset:   {DATASET}")
print(f"MB config: {ALGORITHM} α={ALPHA}")
print(f"Target N:  {TOP_N} features")
print(f"Output:    {OUT_DIR}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv(MERGE_DIR / f"{DATASET}.csv", index_col=0)
df = df.dropna(subset=["OS", "OS.time"])

feat_cols = [c for c in df.columns if c not in ("OS", "OS.time")]
X_all = df[feat_cols]
y_time = df["OS.time"].values
y_event = df["OS"].values

print(f"Shape: {df.shape}")
print(f"Features: {len(feat_cols)}")
print(f"Samples:  {len(df)}")
print(f"Events:   {int(y_event.sum())}")

# Load MB gene list (truly significant ones)
gene_file = MB_DIR / RNA_DS / f"{ALGORITHM}_alpha{ALPHA:.2f}_genes.txt"
if not gene_file.exists():
    gene_file = MB_DIR / RNA_DS / f"{ALGORITHM}_alpha{ALPHA}_genes.txt"

mb_genes_raw = [l.strip() for l in gene_file.read_text().splitlines() if l.strip()]
mb_genes = [f"RNA_{g}" for g in mb_genes_raw if f"RNA_{g}" in feat_cols]
print(f"\nMB selected genes (RNA): {len(mb_genes)}")

# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def _cindex_manual(times, events, scores):
    n_conc = n_disc = n_tied = 0
    for i in range(len(times)):
        for j in range(i+1, len(times)):
            if events[i] == 0 and events[j] == 0: continue
            if times[i] == times[j]: continue
            e, l = (i,j) if times[i] < times[j] else (j,i)
            if not events[e]: continue
            if scores[e] > scores[l]: n_conc += 1
            elif scores[e] < scores[l]: n_disc += 1
            else: n_tied += 1
    total = n_conc + n_disc + n_tied
    return (n_conc + 0.5*n_tied) / total if total > 0 else 0.5


def cv_metrics(df, features, n_folds=N_FOLDS):
    """5-fold CV: C-index (Cox) + AUC-5yr (logistic)."""
    if not features:
        return 0.5, 0.5, 0

    sub = df[features + ["OS", "OS.time"]].dropna()
    if len(sub) < 40:
        return 0.5, 0.5, len(features)

    kf  = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    threshold = SURVIVAL_YEARS * 365

    c_scores, auc_scores = [], []

    for tr_idx, te_idx in kf.split(sub):
        train, test = sub.iloc[tr_idx], sub.iloc[te_idx]

        # C-index via Cox
        try:
            from lifelines import CoxPHFitter
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train, duration_col="OS.time", event_col="OS")
            pred = cph.predict_partial_hazard(test).values
            c_scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, pred))
        except Exception:
            Xtr = StandardScaler().fit(train[features]).transform(train[features])
            Xte = StandardScaler().fit(train[features]).transform(test[features])
            score = Xte.mean(1)
            c_scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, score))

        # AUC-5yr via logistic
        try:
            sub5_tr = train[(train["OS.time"] >= threshold) | (train["OS"]==1)]
            sub5_te = test[(test["OS.time"] >= threshold) | (test["OS"]==1)]
            y_tr = ((sub5_tr["OS"]==1) & (sub5_tr["OS.time"]<=threshold)).astype(int)
            y_te = ((sub5_te["OS"]==1) & (sub5_te["OS.time"]<=threshold)).astype(int)
            if y_tr.sum() >= 3 and (1-y_tr).sum() >= 3 and y_te.sum() >= 1:
                sc = StandardScaler()
                Xtr5 = sc.fit_transform(sub5_tr[features])
                Xte5 = sc.transform(sub5_te[features])
                lr = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED)
                lr.fit(Xtr5, y_tr)
                prob = lr.predict_proba(Xte5)[:,1]
                auc_scores.append(roc_auc_score(y_te, prob))
        except Exception:
            pass

    c_idx = float(np.mean(c_scores)) if c_scores else 0.5
    auc   = float(np.mean(auc_scores)) if auc_scores else 0.5
    return c_idx, auc, len(features)


# ============================================================================
# FEATURE SELECTION STRATEGIES
# ============================================================================

def strategy_0_no_fallback(df, feat_cols, mb_genes, top_n):
    """Only truly MB-significant RNA genes + all CLIN features."""
    clin_cols = [c for c in feat_cols if c.startswith("CLIN_")]
    selected = clin_cols + mb_genes
    return [f for f in selected if f in feat_cols]


def strategy_1_cox_univariate(df, feat_cols, mb_genes, top_n):
    """Top-N features by univariate Cox p-value."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    sub = df[feat_cols + ["OS", "OS.time"]].dropna()
    pvals = {}
    for col in feat_cols:
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(sub[[col, "OS", "OS.time"]], duration_col="OS.time", event_col="OS")
            pvals[col] = cph.summary["p"].iloc[0]
        except Exception:
            pvals[col] = 1.0
    sorted_feats = sorted(pvals, key=lambda x: pvals[x])
    return sorted_feats[:top_n]


def strategy_2_mi_topn(df, feat_cols, mb_genes, top_n):
    """Top-N by mutual information (current baseline fallback)."""
    from sklearn.feature_selection import mutual_info_regression
    sub = df[feat_cols + ["OS.time"]].dropna()
    X = StandardScaler().fit_transform(sub[feat_cols].values)
    mi = mutual_info_regression(X, sub["OS.time"].values, random_state=RANDOM_SEED)
    idx = np.argsort(mi)[::-1][:top_n]
    return [feat_cols[i] for i in idx]


def strategy_3_mi_variance(df, feat_cols, mb_genes, top_n):
    """Top-N by MI × variance (rewards both relevance and variability)."""
    from sklearn.feature_selection import mutual_info_regression
    sub = df[feat_cols + ["OS.time"]].dropna()
    X_raw = sub[feat_cols].values
    X_sc  = StandardScaler().fit_transform(X_raw)
    mi  = mutual_info_regression(X_sc, sub["OS.time"].values, random_state=RANDOM_SEED)
    var = X_raw.var(axis=0)
    var = var / (var.max() + 1e-8)
    score = mi * (1 + var)
    idx = np.argsort(score)[::-1][:top_n]
    return [feat_cols[i] for i in idx]


def strategy_4_elastic_net_cox(df, feat_cols, mb_genes, top_n):
    """Elastic Net Cox — features with non-zero coefficients."""
    try:
        from lifelines import CoxPHFitter
        from sklearn.linear_model import ElasticNet
        sub = df[feat_cols + ["OS", "OS.time"]].dropna()
        X = StandardScaler().fit_transform(sub[feat_cols].values)
        # Use OS.time * OS as pseudo-response for EN
        y = sub["OS.time"].values * sub["OS"].values
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=RANDOM_SEED)
        en.fit(X, y)
        nonzero = np.where(np.abs(en.coef_) > 1e-6)[0]
        if len(nonzero) == 0:
            nonzero = np.argsort(np.abs(en.coef_))[::-1][:top_n]
        elif len(nonzero) > top_n:
            # Keep top_n by coefficient magnitude
            nonzero = sorted(nonzero, key=lambda i: abs(en.coef_[i]), reverse=True)[:top_n]
        return [feat_cols[i] for i in nonzero]
    except Exception as e:
        print(f"    ElasticNet failed: {e}")
        return strategy_2_mi_topn(df, feat_cols, mb_genes, top_n)


def strategy_5_mi_strict(df, feat_cols, mb_genes, top_n):
    """Strict MI permutation test (alpha=0.01, no padding) — truly significant only."""
    from sklearn.feature_selection import mutual_info_regression
    rng = np.random.default_rng(RANDOM_SEED)
    sub = df[feat_cols + ["OS.time"]].dropna()
    X = StandardScaler().fit_transform(sub[feat_cols].values)
    y = sub["OS.time"].values
    mi = mutual_info_regression(X, y, random_state=RANDOM_SEED)
    # Permutation null (100 permutations, strict alpha=0.01)
    N_PERM = 100
    null = np.array([
        mutual_info_regression(X, rng.permutation(y), random_state=p)
        for p in range(N_PERM)
    ])
    threshold = np.percentile(null, 99, axis=0)  # alpha=0.01
    sig = np.where(mi > threshold)[0].tolist()
    if not sig:
        return []  # no fallback — truly empty
    return [feat_cols[i] for i in sorted(sig, key=lambda i: mi[i], reverse=True)]


STRATEGIES = {
    "S0_no_fallback":      strategy_0_no_fallback,
    "S1_cox_univariate":   strategy_1_cox_univariate,
    "S2_mi_top20":         strategy_2_mi_topn,
    "S3_mi_variance":      strategy_3_mi_variance,
    "S4_elastic_net_cox":  strategy_4_elastic_net_cox,
    "S5_mi_strict_no_pad": strategy_5_mi_strict,
}

# ============================================================================
# RUN EXPERIMENT
# ============================================================================
print("\n" + "="*70)
print(f"EXPERIMENT: {len(STRATEGIES)} strategies × TOP_N={TOP_N}")
print("="*70)

results = []

for name, fn in STRATEGIES.items():
    print(f"\n[{name}]")
    t0 = time.time()
    try:
        selected = fn(df, feat_cols, mb_genes, TOP_N)
    except Exception as e:
        print(f"  ERROR in selection: {e}")
        selected = []

    elapsed_sel = time.time() - t0
    print(f"  Selected: {len(selected)} features  ({elapsed_sel:.1f}s)")

    if selected:
        # Modality breakdown
        mod_counts = {}
        for prefix in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]:
            n = sum(1 for f in selected if f.startswith(prefix))
            if n: mod_counts[prefix.rstrip("_")] = n
        print(f"  Modalities: {mod_counts}")

    t1 = time.time()
    c_idx, auc5, n_feat = cv_metrics(df, selected)
    elapsed_cv = time.time() - t1
    print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}  ({elapsed_cv:.1f}s CV)")

    results.append({
        "strategy":   name,
        "n_features": n_feat,
        "c_index":    round(c_idx, 4),
        "auc_5yr":    round(auc5, 4),
        "features":   selected,
    })

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY — sorted by C-index")
print("="*70)

res_df = pd.DataFrame([{k:v for k,v in r.items() if k!="features"} for r in results])
res_df = res_df.sort_values(["c_index","auc_5yr"], ascending=False).reset_index(drop=True)
print(res_df.to_string(index=False))

res_df.to_csv(OUT_DIR / "fallback_experiment_results.csv", index=False)

# Save best strategy features
best_name = res_df.iloc[0]["strategy"]
best_feat = next(r["features"] for r in results if r["strategy"] == best_name)
with open(OUT_DIR / "best_strategy_features.txt", "w") as f:
    f.write(f"Best strategy: {best_name}\n")
    f.write(f"C-index: {res_df.iloc[0]['c_index']}\n")
    f.write(f"AUC-5yr: {res_df.iloc[0]['auc_5yr']}\n\n")
    f.write("\n".join(best_feat))

print(f"\nBest: {best_name}  C-index={res_df.iloc[0]['c_index']}  AUC={res_df.iloc[0]['auc_5yr']}")
print(f"\nSaved to: {OUT_DIR}")
