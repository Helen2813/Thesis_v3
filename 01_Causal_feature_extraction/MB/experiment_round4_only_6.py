"""
ROUND 4 ONLY: Fine-tuning around S30 (cox_pen10_top20, C-index=0.8075)
Dataset: 08_composite | IAMB α=0.05
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

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    cwd = Path.cwd()
    SCRIPT_DIR = cwd if cwd.name == "MB" else cwd / "MB"

MERGE_DIR   = SCRIPT_DIR.parent / "MERGE_continuous_outer"
MB_DIR      = SCRIPT_DIR.parent / "RNA" / "mb_results"
OUT_DIR     = SCRIPT_DIR / "results_fallback_experiment"
OUT_DIR.mkdir(exist_ok=True)

DATASET     = "08_composite"
RNA_DS      = "rna_8_composite_1000genes"
ALGORITHM   = "IAMB"
ALPHA       = 0.05
N_FOLDS     = 5
RANDOM_SEED = 42
SURVIVAL_YEARS = 5
TOP_N       = 20

print(f"Dataset:   {DATASET}")
print(f"MB config: {ALGORITHM} α={ALPHA}")
print(f"Output:    {OUT_DIR}")

df = pd.read_csv(MERGE_DIR / f"{DATASET}.csv", index_col=0)
df = df.dropna(subset=["OS", "OS.time"])
feat_cols = [c for c in df.columns if c not in ("OS", "OS.time")]

gene_file = MB_DIR / RNA_DS / f"{ALGORITHM}_alpha{ALPHA:.2f}_genes.txt"
mb_genes = [f"RNA_{l.strip()}" for l in gene_file.read_text().splitlines()
            if l.strip() and f"RNA_{l.strip()}" in feat_cols]

print(f"Samples: {len(df)}  Events: {int(df['OS'].sum())}  Features: {len(feat_cols)}")
print(f"MB genes: {len(mb_genes)}")

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
    if not features: return 0.5, 0.5, 0
    sub = df[features+["OS","OS.time"]].dropna()
    if len(sub)<40: return 0.5, 0.5, len(features)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    threshold = SURVIVAL_YEARS*365
    c_scores, auc_scores = [], []
    for tr,te in kf.split(sub):
        train,test = sub.iloc[tr], sub.iloc[te]
        try:
            from lifelines import CoxPHFitter
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train, duration_col="OS.time", event_col="OS")
            pred = cph.predict_partial_hazard(test).values
            c_scores.append(_cindex_manual(test["OS.time"].values,test["OS"].values,pred))
        except Exception:
            X = StandardScaler().fit(train[features]).transform(test[features])
            c_scores.append(_cindex_manual(test["OS.time"].values,test["OS"].values,X.mean(1)))
        try:
            s5tr = train[(train["OS.time"]>=threshold)|(train["OS"]==1)]
            s5te = test[(test["OS.time"]>=threshold)|(test["OS"]==1)]
            ytr = ((s5tr["OS"]==1)&(s5tr["OS.time"]<=threshold)).astype(int)
            yte = ((s5te["OS"]==1)&(s5te["OS.time"]<=threshold)).astype(int)
            if ytr.sum()>=3 and (1-ytr).sum()>=3 and yte.sum()>=1:
                sc = StandardScaler()
                lr = LogisticRegression(C=0.1,max_iter=500,random_state=RANDOM_SEED)
                lr.fit(sc.fit_transform(s5tr[features]),ytr)
                auc_scores.append(roc_auc_score(yte,lr.predict_proba(sc.transform(s5te[features]))[:,1]))
        except Exception: pass
    return float(np.mean(c_scores)) if c_scores else 0.5, float(np.mean(auc_scores)) if auc_scores else 0.5, len(features)

def cox_topn(df, feat_cols, n, penalizer=0.1):
    from lifelines import CoxPHFitter
    sub = df[feat_cols+["OS","OS.time"]].dropna()
    pvals = {}
    for col in feat_cols:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(sub[[col,"OS","OS.time"]],duration_col="OS.time",event_col="OS")
            pvals[col] = cph.summary["p"].iloc[0]
        except Exception: pvals[col] = 1.0
    return sorted(pvals,key=lambda x:pvals[x])[:n]

def strategy_5_mi_strict(df, feat_cols, mb_genes, top_n):
    from sklearn.feature_selection import mutual_info_regression
    rng = np.random.default_rng(RANDOM_SEED)
    sub = df[feat_cols+["OS.time"]].dropna()
    X = StandardScaler().fit_transform(sub[feat_cols].values)
    y = sub["OS.time"].values
    mi = mutual_info_regression(X,y,random_state=RANDOM_SEED)
    null = np.array([mutual_info_regression(X,rng.permutation(y),random_state=p) for p in range(100)])
    thresh = np.percentile(null,99,axis=0)
    sig = np.where(mi>thresh)[0].tolist()
    if not sig: return []
    return [feat_cols[i] for i in sorted(sig,key=lambda i:mi[i],reverse=True)]

def _cox_rerank_auc(df, feat_cols, n):
    from sklearn.metrics import roc_auc_score
    candidates = cox_topn(df,feat_cols,40,penalizer=10.0)
    threshold = SURVIVAL_YEARS*365
    sub = df[candidates+["OS","OS.time"]].dropna()
    sub5 = sub[(sub["OS.time"]>=threshold)|(sub["OS"]==1)].copy()
    y5 = ((sub5["OS"]==1)&(sub5["OS.time"]<=threshold)).astype(int)
    aucs = {}
    for col in candidates:
        try: aucs[col] = roc_auc_score(y5,sub5[col])
        except: aucs[col]=0.5
    return sorted(aucs,key=lambda x:aucs[x],reverse=True)[:n]

# ============================================================================
# ROUND 4: FINE-TUNING AROUND S30 (penalizer + N variants)
# ============================================================================
print("\n" + "="*70)
print("ROUND 4: FINE-TUNING AROUND S30 (best: cox_pen10_top20)")
print("="*70)

ROUND4 = {
    # Penalizer variants around 10
    "S34_cox_pen5_top20":   lambda df,f,m,n: cox_topn(df,f,20,penalizer=5.0),
    "S35_cox_pen20_top20":  lambda df,f,m,n: cox_topn(df,f,20,penalizer=20.0),
    "S36_cox_pen50_top20":  lambda df,f,m,n: cox_topn(df,f,20,penalizer=50.0),
    "S37_cox_pen100_top20": lambda df,f,m,n: cox_topn(df,f,20,penalizer=100.0),
    "S38_cox_pen3_top20":   lambda df,f,m,n: cox_topn(df,f,20,penalizer=3.0),
    "S39_cox_pen7_top20":   lambda df,f,m,n: cox_topn(df,f,20,penalizer=7.0),
    "S40_cox_pen15_top20":  lambda df,f,m,n: cox_topn(df,f,20,penalizer=15.0),

    # N variants with penalizer=10
    "S41_cox_pen10_top15":  lambda df,f,m,n: cox_topn(df,f,15,penalizer=10.0),
    "S42_cox_pen10_top25":  lambda df,f,m,n: cox_topn(df,f,25,penalizer=10.0),
    "S43_cox_pen10_top30":  lambda df,f,m,n: cox_topn(df,f,30,penalizer=10.0),
    "S44_cox_pen10_top12":  lambda df,f,m,n: cox_topn(df,f,12,penalizer=10.0),
    "S45_cox_pen10_top18":  lambda df,f,m,n: cox_topn(df,f,18,penalizer=10.0),

    # pen10 + per modality
    "S46_cox_pen10_per_mod5": lambda df,f,m,n: sum([
        cox_topn(df,[c for c in f if c.startswith(p)],5,penalizer=10.0)
        for p in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]
        if any(c.startswith(p) for c in f)], []),
    "S47_cox_pen10_per_mod3": lambda df,f,m,n: sum([
        cox_topn(df,[c for c in f if c.startswith(p)],3,penalizer=10.0)
        for p in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]
        if any(c.startswith(p) for c in f)], []),

    # pen10 clin+omics split
    "S48_pen10_clin10_omics10": lambda df,f,m,n: list(set(
        cox_topn(df,[c for c in f if c.startswith("CLIN_")],10,penalizer=10.0) +
        cox_topn(df,[c for c in f if not c.startswith("CLIN_")],10,penalizer=10.0))),
    "S49_pen10_clin15_omics10": lambda df,f,m,n: list(set(
        cox_topn(df,[c for c in f if c.startswith("CLIN_")],15,penalizer=10.0) +
        cox_topn(df,[c for c in f if not c.startswith("CLIN_")],10,penalizer=10.0))),
    "S50_pen10_clin5_omics15":  lambda df,f,m,n: list(set(
        cox_topn(df,[c for c in f if c.startswith("CLIN_")],5,penalizer=10.0) +
        cox_topn(df,[c for c in f if not c.startswith("CLIN_")],15,penalizer=10.0))),

    # S30 + ensemble with strict MI
    "S51_pen10_union_mi":   lambda df,f,m,n: list(set(
        cox_topn(df,f,20,penalizer=10.0) +
        (strategy_5_mi_strict(df,f,m,n) or []))),
    "S52_pen10_intersect_mi": lambda df,f,m,n: (lambda s30,s5:
        cox_topn(df, list(set(s30)&set(s5)) or s30, min(20,len(set(s30)&set(s5)) or 20),
                 penalizer=10.0))(
        cox_topn(df,f,40,penalizer=10.0),
        strategy_5_mi_strict(df,f,m,n) or []),

    # pen10 excluding missingness indicators
    "S53_pen10_no_miss": lambda df,f,m,n: cox_topn(
        df,[c for c in f if not c.endswith("_missing")],20,penalizer=10.0),

    # pen10 reranked by AUC-5yr
    "S54_pen10_rerank_auc": lambda df,f,m,n: _cox_rerank_auc(
        df, cox_topn(df,f,40,penalizer=10.0), 20),
}

results4 = []

for name, fn in ROUND4.items():
    print(f"\n[{name}]")
    t0 = time.time()
    try:
        selected = fn(df, feat_cols, mb_genes, TOP_N)
        selected = [f for f in selected if f in feat_cols]
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

    results4.append({
        "strategy":   name,
        "n_features": n_feat,
        "c_index":    round(c_idx, 4),
        "auc_5yr":    round(auc5, 4),
        "features":   selected,
    })

# Final table
all_r = results4
final2 = pd.DataFrame([{k:v for k,v in r.items() if k!="features"} for r in all_r])
final2 = final2.sort_values(["c_index","auc_5yr"], ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("TOP 15 OVERALL")
print("="*70)
print(final2.head(15).to_string(index=False))

final2.to_csv(OUT_DIR / "fallback_experiment_all.csv", index=False)

best_name = final2.iloc[0]["strategy"]
best_feat = next(r["features"] for r in all_r if r["strategy"] == best_name)
with open(OUT_DIR / "best_strategy_features.txt", "w") as f:
    f.write(f"Best strategy: {best_name}\n")
    f.write(f"C-index: {final2.iloc[0]['c_index']}\n")
    f.write(f"AUC-5yr: {final2.iloc[0]['auc_5yr']}\n\n")
    f.write("\n".join(best_feat))

print(f"\nBest: {best_name}")
print(f"  C-index: {final2.iloc[0]['c_index']}")
print(f"  AUC-5yr: {final2.iloc[0]['auc_5yr']}")
print(f"\nSaved: fallback_experiment_all.csv")
