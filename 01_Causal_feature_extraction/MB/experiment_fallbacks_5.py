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


# ============================================================================
# ROUND 2: ADDITIONAL STRATEGIES
# ============================================================================
print("\n" + "="*70)
print("ROUND 2: ADDITIONAL STRATEGIES (tuning S1 + ensemble)")
print("="*70)

def strategy_s1_n(df, feat_cols, mb_genes, n):
    """Cox univariate top-N."""
    from lifelines import CoxPHFitter
    sub = df[feat_cols + ["OS", "OS.time"]].dropna()
    pvals = {}
    for col in feat_cols:
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(sub[[col, "OS", "OS.time"]], duration_col="OS.time", event_col="OS")
            pvals[col] = cph.summary["p"].iloc[0]
        except Exception:
            pvals[col] = 1.0
    return sorted(pvals, key=lambda x: pvals[x])[:n]


def strategy_s6_cox15(df, feat_cols, mb_genes, top_n):
    """Cox univariate top-15."""
    return strategy_s1_n(df, feat_cols, mb_genes, 15)

def strategy_s7_cox30(df, feat_cols, mb_genes, top_n):
    """Cox univariate top-30."""
    return strategy_s1_n(df, feat_cols, mb_genes, 30)

def strategy_s8_ensemble_union(df, feat_cols, mb_genes, top_n):
    """Union of S1(top-15) + S5(strict MI) — capped at top_n by Cox p-value."""
    s1 = set(strategy_s1_n(df, feat_cols, mb_genes, 15))
    s5 = set(strategy_5_mi_strict(df, feat_cols, mb_genes, top_n))
    union = list(s1 | s5)
    # Rank union by Cox p-value and take top_n
    return strategy_s1_n(df, union, mb_genes, min(top_n, len(union))) if union else []

def strategy_s9_ensemble_intersect(df, feat_cols, mb_genes, top_n):
    """Intersection of Cox top-30 and strict MI — features both agree on."""
    s1 = set(strategy_s1_n(df, feat_cols, mb_genes, 30))
    s5 = set(strategy_5_mi_strict(df, feat_cols, mb_genes, top_n))
    inter = list(s1 & s5)
    if not inter:
        print("    Intersection empty — falling back to Cox top-20")
        return strategy_s1_n(df, feat_cols, mb_genes, 20)
    return inter

def strategy_s10_cox_mb_hybrid(df, feat_cols, mb_genes, top_n):
    """Hybrid: MB-selected RNA genes + Cox top-10 from non-RNA features."""
    non_rna = [c for c in feat_cols if not c.startswith("RNA_")]
    cox_non_rna = strategy_s1_n(df, non_rna, mb_genes, 10)
    # Add MB RNA genes (already selected by causal method)
    rna_in_feat = [g for g in mb_genes if g in feat_cols]
    combined = list(set(cox_non_rna + rna_in_feat))
    return combined

ROUND2_STRATEGIES = {
    "S6_cox_top15":         strategy_s6_cox15,
    "S7_cox_top30":         strategy_s7_cox30,
    "S8_ensemble_union":    strategy_s8_ensemble_union,
    "S9_ensemble_intersect":strategy_s9_ensemble_intersect,
    "S10_cox_mb_hybrid":    strategy_s10_cox_mb_hybrid,
}

results2 = []

for name, fn in ROUND2_STRATEGIES.items():
    print(f"\n[{name}]")
    t0 = time.time()
    try:
        selected = fn(df, feat_cols, mb_genes, TOP_N)
    except Exception as e:
        print(f"  ERROR: {e}")
        selected = []
    elapsed_sel = time.time() - t0

    print(f"  Selected: {len(selected)} features  ({elapsed_sel:.1f}s)")
    if selected:
        mod_counts = {}
        for prefix in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]:
            n = sum(1 for f in selected if f.startswith(prefix))
            if n: mod_counts[prefix.rstrip("_")] = n
        print(f"  Modalities: {mod_counts}")

    c_idx, auc5, n_feat = cv_metrics(df, selected)
    print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}")

    results2.append({
        "strategy":   name,
        "n_features": n_feat,
        "c_index":    round(c_idx, 4),
        "auc_5yr":    round(auc5, 4),
        "features":   selected,
    })

# Combined summary
all_results = results + results2
all_df = pd.DataFrame([{k:v for k,v in r.items() if k!="features"} for r in all_results])
all_df = all_df.sort_values(["c_index","auc_5yr"], ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("FINAL SUMMARY — ALL STRATEGIES")
print("="*70)
print(all_df.to_string(index=False))

all_df.to_csv(OUT_DIR / "fallback_experiment_all.csv", index=False)

best_name = all_df.iloc[0]["strategy"]
best_feat = next(r["features"] for r in all_results if r["strategy"] == best_name)
with open(OUT_DIR / "best_strategy_features.txt", "w") as f:
    f.write(f"Best strategy: {best_name}\n")
    f.write(f"C-index: {all_df.iloc[0]['c_index']}\n")
    f.write(f"AUC-5yr: {all_df.iloc[0]['auc_5yr']}\n\n")
    f.write("\n".join(best_feat))

print(f"\nBest overall: {best_name}")
print(f"  C-index: {all_df.iloc[0]['c_index']}")
print(f"  AUC-5yr: {all_df.iloc[0]['auc_5yr']}")
print(f"\nSaved: fallback_experiment_all.csv + best_strategy_features.txt")


# ============================================================================
# ROUND 3: MANY MORE COX VARIANTS + OTHER APPROACHES
# ============================================================================
print("\n" + "="*70)
print("ROUND 3: MORE VARIANTS")
print("="*70)

def cox_topn(df, feat_cols, n, penalizer=0.1):
    """Cox univariate top-N with given penalizer."""
    from lifelines import CoxPHFitter
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

def cox_significant(df, feat_cols, alpha_thresh=0.05, penalizer=0.1):
    """Only features with Cox p < alpha_thresh."""
    from lifelines import CoxPHFitter
    sub = df[feat_cols + ["OS", "OS.time"]].dropna()
    sig = []
    for col in feat_cols:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(sub[[col,"OS","OS.time"]], duration_col="OS.time", event_col="OS")
            if cph.summary["p"].iloc[0] < alpha_thresh:
                sig.append((col, cph.summary["p"].iloc[0]))
        except Exception:
            pass
    return [c for c,p in sorted(sig, key=lambda x: x[1])]

ROUND3 = {
    # Cox top-N variants
    "S11_cox_top10":   lambda df,f,m,n: cox_topn(df,f,10),
    "S12_cox_top25":   lambda df,f,m,n: cox_topn(df,f,25),
    "S13_cox_top35":   lambda df,f,m,n: cox_topn(df,f,35),
    "S14_cox_top40":   lambda df,f,m,n: cox_topn(df,f,40),
    "S15_cox_top50":   lambda df,f,m,n: cox_topn(df,f,50),

    # Cox significant only (different alpha thresholds)
    "S16_cox_p05":     lambda df,f,m,n: cox_significant(df,f,0.05),
    "S17_cox_p10":     lambda df,f,m,n: cox_significant(df,f,0.10),
    "S18_cox_p20":     lambda df,f,m,n: cox_significant(df,f,0.20),
    "S19_cox_p01":     lambda df,f,m,n: cox_significant(df,f,0.01),

    # Cox per modality: top-K from each separately then combine
    "S20_cox_per_mod_3": lambda df,f,m,n: sum([
        cox_topn(df,[c for c in f if c.startswith(p)],3)
        for p in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]
        if any(c.startswith(p) for c in f)], []),
    "S21_cox_per_mod_5": lambda df,f,m,n: sum([
        cox_topn(df,[c for c in f if c.startswith(p)],5)
        for p in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]
        if any(c.startswith(p) for c in f)], []),
    "S22_cox_per_mod_7": lambda df,f,m,n: sum([
        cox_topn(df,[c for c in f if c.startswith(p)],7)
        for p in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]
        if any(c.startswith(p) for c in f)], []),

    # Cox + MB RNA hybrid with different N
    "S23_cox10_mb_rna":  lambda df,f,m,n: list(set(
        cox_topn(df,[c for c in f if not c.startswith("RNA_")],10) +
        [g for g in m if g in f][:20])),
    "S24_cox15_mb_rna":  lambda df,f,m,n: list(set(
        cox_topn(df,[c for c in f if not c.startswith("RNA_")],15) +
        [g for g in m if g in f][:20])),
    "S25_cox5_mb_rna":   lambda df,f,m,n: list(set(
        cox_topn(df,[c for c in f if not c.startswith("RNA_")],5) +
        [g for g in m if g in f][:20])),

    # Cox only on specific modalities
    "S26_cox_clin_only": lambda df,f,m,n: cox_topn(df,[c for c in f if c.startswith("CLIN_")],20),
    "S27_cox_omics_only":lambda df,f,m,n: cox_topn(df,[c for c in f if not c.startswith("CLIN_")],20),
    "S28_cox_clin10_omics10": lambda df,f,m,n: list(set(
        cox_topn(df,[c for c in f if c.startswith("CLIN_")],10) +
        cox_topn(df,[c for c in f if not c.startswith("CLIN_")],10))),

    # Cox with stricter penalizer (more regularization)
    "S29_cox_pen01_top20": lambda df,f,m,n: cox_topn(df,f,20,penalizer=0.01),
    "S30_cox_pen10_top20": lambda df,f,m,n: cox_topn(df,f,20,penalizer=1.0),

    # Random Survival Forest feature importance (if installed)
    "S31_rsf_top20": lambda df,f,m,n: _rsf_topn(df,f,20),

    # Cox + missingness indicators prioritized
    "S32_cox_no_missing_ind": lambda df,f,m,n: cox_topn(
        df,[c for c in f if not c.endswith("_missing")],20),

    # Cox top-20 then reranked by AUC-5yr individually
    "S33_cox20_rerank_auc": lambda df,f,m,n: _cox_rerank_auc(df,f,20),
}

def _rsf_topn(df, feat_cols, n):
    """Random Survival Forest feature importance top-N."""
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        sub = df[feat_cols + ["OS","OS.time"]].dropna()
        X = StandardScaler().fit_transform(sub[feat_cols].values)
        y = Surv.from_dataframe("OS","OS.time", sub)
        rsf = RandomSurvivalForest(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        rsf.fit(X, y)
        idx = np.argsort(rsf.feature_importances_)[::-1][:n]
        return [feat_cols[i] for i in idx]
    except Exception as e:
        print(f"    RSF failed ({e}), falling back to Cox top-{n}")
        return cox_topn(df, feat_cols, n)

def _cox_rerank_auc(df, feat_cols, n):
    """Take Cox top-20, rerank by individual AUC-5yr, return top-N."""
    from sklearn.metrics import roc_auc_score
    candidates = cox_topn(df, feat_cols, 40)
    threshold = SURVIVAL_YEARS * 365
    sub = df[candidates + ["OS","OS.time"]].dropna()
    sub5 = sub[(sub["OS.time"] >= threshold) | (sub["OS"]==1)].copy()
    y5 = ((sub5["OS"]==1) & (sub5["OS.time"]<=threshold)).astype(int)
    aucs = {}
    for col in candidates:
        try:
            aucs[col] = roc_auc_score(y5, sub5[col])
        except Exception:
            aucs[col] = 0.5
    return sorted(aucs, key=lambda x: aucs[x], reverse=True)[:n]

results3 = []

for name, fn in ROUND3.items():
    print(f"\n[{name}]")
    t0 = time.time()
    try:
        selected = fn(df, feat_cols, mb_genes, TOP_N)
        selected = [f for f in selected if f in feat_cols]  # ensure valid
    except Exception as e:
        print(f"  ERROR: {e}")
        selected = []
    elapsed = time.time() - t0

    print(f"  Selected: {len(selected)} features  ({elapsed:.1f}s)")
    if selected:
        mod_counts = {}
        for prefix in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]:
            n = sum(1 for f in selected if f.startswith(prefix))
            if n: mod_counts[prefix.rstrip("_")] = n
        print(f"  Modalities: {mod_counts}")

    c_idx, auc5, n_feat = cv_metrics(df, selected)
    print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}")

    results3.append({
        "strategy":   name,
        "n_features": n_feat,
        "c_index":    round(c_idx, 4),
        "auc_5yr":    round(auc5, 4),
        "features":   selected,
    })

# Final combined summary
all_results_final = results + results2 + results3
final_df = pd.DataFrame([{k:v for k,v in r.items() if k!="features"} for r in all_results_final])
final_df = final_df.sort_values(["c_index","auc_5yr"], ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("FINAL SUMMARY — ALL ROUNDS")
print("="*70)
print(final_df.to_string(index=False))

final_df.to_csv(OUT_DIR / "fallback_experiment_all.csv", index=False)

best_name = final_df.iloc[0]["strategy"]
best_feat = next(r["features"] for r in all_results_final if r["strategy"] == best_name)
with open(OUT_DIR / "best_strategy_features.txt", "w") as f:
    f.write(f"Best strategy: {best_name}\n")
    f.write(f"C-index: {final_df.iloc[0]['c_index']}\n")
    f.write(f"AUC-5yr: {final_df.iloc[0]['auc_5yr']}\n\n")
    f.write("\n".join(best_feat))

print(f"\nBest: {best_name}  C-index={final_df.iloc[0]['c_index']}  AUC={final_df.iloc[0]['auc_5yr']}")
print(f"Saved: {OUT_DIR}/fallback_experiment_all.csv")
