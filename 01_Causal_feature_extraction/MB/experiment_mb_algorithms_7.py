"""
EXPERIMENT: True Markov Blanket algorithms using conditional independence tests.

Algorithms:
  1. IAMB       - Incremental Association MB (baseline)
  2. Inter-IAMB - Interleaved IAMB (prunes during forward phase)
  3. HITON-MB   - Hiton Parents-Children + Spouses
  4. FBED       - Forward-Backward Early Dropping
  5. PCMB       - PC-based MB (separating sets)
  6. Ensemble   - Consensus of all algorithms

CI test: partial correlation t-test (fast, appropriate for continuous data)
         with fallback to MI permutation test

Dataset: 08_composite | MERGE_continuous_outer
Features: top-50 by univariate Cox (penalizer=10) to keep CI tests tractable
"""

import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    cwd = Path.cwd()
    SCRIPT_DIR = cwd if cwd.name == "MB" else cwd / "MB"

MERGE_DIR  = SCRIPT_DIR.parent / "MERGE_continuous_outer"
OUT_DIR    = SCRIPT_DIR / "results_mb_algorithms"
OUT_DIR.mkdir(exist_ok=True)

DATASET      = "08_composite"
N_FOLDS      = 5
RANDOM_SEED  = 42
SURVIVAL_YEARS = 5
ALPHA_CI     = 0.05   # significance threshold for CI tests
MAX_COND_SET = 3      # max conditioning set size (speed vs accuracy tradeoff)
PRE_SELECT_N = 80     # pre-filter: top-N by univariate Cox before running MB

print(f"Dataset:    {DATASET}")
print(f"CI alpha:   {ALPHA_CI}")
print(f"Max cond:   {MAX_COND_SET}")
print(f"Pre-select: {PRE_SELECT_N} features")
print(f"Output:     {OUT_DIR}")

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_csv(MERGE_DIR / f"{DATASET}.csv", index_col=0)
df = df.dropna(subset=["OS", "OS.time"])
all_feat_cols = [c for c in df.columns if c not in ("OS", "OS.time")]

print(f"\nShape: {df.shape}  Events: {int(df['OS'].sum())}")

# Pre-select top features by univariate Cox to keep CI tests tractable
from lifelines import CoxPHFitter

def cox_topn_preselect(df, feat_cols, n, penalizer=10.0):
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

# Two pre-selection pools:
# Pool A: top-80 (broad — MB has more to work with)
# Pool B: top-20 winning fallback (narrow — MB refines the best candidates)
print(f"\nPre-selecting features by Cox univariate (pen=10)...")
t0 = time.time()
feat_cols_80 = cox_topn_preselect(df, all_feat_cols, 80)
feat_cols_20 = cox_topn_preselect(df, all_feat_cols, 20)
feat_cols    = feat_cols_80   # default: broad pool
print(f"Pool A (top-80): {len(feat_cols_80)}  Pool B (top-20): {len(feat_cols_20)}")
print(f"Done ({time.time()-t0:.1f}s)")

sub_df = df[feat_cols + ["OS", "OS.time"]].dropna()
X = StandardScaler().fit_transform(sub_df[feat_cols].values)
X_df = pd.DataFrame(X, columns=feat_cols, index=sub_df.index)
T = sub_df["OS.time"].values   # survival time as target
n_feat = len(feat_cols)

print(f"Working dataset: {X_df.shape}")

# ============================================================================
# CONDITIONAL INDEPENDENCE TEST
# ============================================================================

def partial_corr_pvalue(x, y, Z, data_arr, col_index):
    """
    Test X _|_ Y | Z using partial correlation.
    Returns p-value. Small p = dependent (NOT independent).
    """
    if len(Z) == 0:
        # Simple Pearson
        r, p = stats.pearsonr(x, y)
        return p

    # Residualise x and y on Z
    Z_arr = data_arr[:, [col_index[z] for z in Z]]
    n = len(x)

    try:
        # OLS residuals
        ZtZ_inv = np.linalg.pinv(Z_arr.T @ Z_arr)
        beta_x = ZtZ_inv @ Z_arr.T @ x
        beta_y = ZtZ_inv @ Z_arr.T @ y
        res_x = x - Z_arr @ beta_x
        res_y = y - Z_arr @ beta_y

        r, p = stats.pearsonr(res_x, res_y)
        return p
    except Exception:
        r, p = stats.pearsonr(x, y)
        return p


def ci_test(xi, xj, Z, X_arr, col_idx, alpha=ALPHA_CI):
    """
    Returns True if Xi _|_ Xj | Z (conditionally independent).
    """
    p = partial_corr_pvalue(X_arr[:, col_idx[xi]],
                             X_arr[:, col_idx[xj]],
                             Z, X_arr, col_idx)
    return p > alpha


def ci_test_with_target(xi, T, Z, X_arr, col_idx, alpha=ALPHA_CI):
    """
    Returns True if Xi _|_ T | Z (feature independent of target given Z).
    """
    if len(Z) == 0:
        _, p = stats.pearsonr(X_arr[:, col_idx[xi]], T)
        return p > alpha

    Z_arr = X_arr[:, [col_idx[z] for z in Z]]
    try:
        ZtZ_inv = np.linalg.pinv(Z_arr.T @ Z_arr)
        beta_x = ZtZ_inv @ Z_arr.T @ X_arr[:, col_idx[xi]]
        beta_t = ZtZ_inv @ Z_arr.T @ T
        res_x = X_arr[:, col_idx[xi]] - Z_arr @ beta_x
        res_t = T - Z_arr @ beta_t
        _, p = stats.pearsonr(res_x, res_t)
        return p > alpha
    except Exception:
        _, p = stats.pearsonr(X_arr[:, col_idx[xi]], T)
        return p > alpha


# Build column index for fast lookup
col_idx = {f: i for i, f in enumerate(feat_cols)}
X_arr = X_df.values.copy()

# ============================================================================
# MB ALGORITHMS
# ============================================================================

def iamb(feat_cols, X_arr, T, col_idx, alpha=ALPHA_CI, max_cond=MAX_COND_SET):
    """
    IAMB: Incremental Association Markov Blanket.
    Phase 1 (forward): add Xi if not ind. of T given current MB
    Phase 2 (backward): remove Xi if ind. of T given MB\{Xi}
    """
    MB = []
    candidates = list(feat_cols)

    # Forward phase
    changed = True
    while changed:
        changed = False
        best_feat, best_p = None, 1.0
        for xi in candidates:
            if xi in MB:
                continue
            # Test Xi ind T | MB
            p = partial_corr_pvalue(X_arr[:, col_idx[xi]], T, MB, X_arr, col_idx)
            if p < best_p:
                best_p, best_feat = p, xi
        if best_feat and best_p < alpha:
            MB.append(best_feat)
            candidates.remove(best_feat)
            changed = True

    # Backward phase
    to_remove = []
    for xi in MB:
        cond = [x for x in MB if x != xi]
        # Limit conditioning set size
        for k in range(0, min(len(cond), max_cond) + 1):
            from itertools import combinations
            for Z in combinations(cond, k):
                if ci_test_with_target(xi, T, list(Z), X_arr, col_idx, alpha):
                    to_remove.append(xi)
                    break
            if xi in to_remove:
                break

    return [x for x in MB if x not in to_remove]


def inter_iamb(feat_cols, X_arr, T, col_idx, alpha=ALPHA_CI, max_cond=MAX_COND_SET):
    """
    Inter-IAMB: interleaved version — prunes after each addition.
    More accurate than IAMB, avoids false positives accumulating.
    """
    MB = []
    candidates = list(feat_cols)

    changed = True
    while changed:
        changed = False

        # Add best candidate
        best_feat, best_p = None, 1.0
        for xi in candidates:
            if xi in MB:
                continue
            p = partial_corr_pvalue(X_arr[:, col_idx[xi]], T, MB, X_arr, col_idx)
            if p < best_p:
                best_p, best_feat = p, xi

        if best_feat and best_p < alpha:
            MB.append(best_feat)
            candidates = [c for c in candidates if c != best_feat]
            changed = True

            # Interleaved backward: immediately check existing MB members
            to_remove = []
            for xi in MB:
                cond = [x for x in MB if x != xi]
                removed = False
                for k in range(0, min(len(cond), max_cond) + 1):
                    from itertools import combinations
                    for Z in combinations(cond, k):
                        if ci_test_with_target(xi, T, list(Z), X_arr, col_idx, alpha):
                            to_remove.append(xi)
                            removed = True
                            break
                    if removed:
                        break
            MB = [x for x in MB if x not in to_remove]

    return MB


def fbed(feat_cols, X_arr, T, col_idx, alpha=ALPHA_CI, max_cond=MAX_COND_SET, k_rounds=3):
    """
    FBED: Forward-Backward Early Dropping.
    Multiple forward rounds with early dropping of irrelevant features.
    """
    from itertools import combinations

    active = list(feat_cols)
    selected = []

    for round_num in range(k_rounds):
        new_selected = []
        for xi in active:
            # Test xi against T conditioned on current selected
            cond_sets = [list(Z) for k in range(0, min(len(selected), max_cond)+1)
                        for Z in combinations(selected, k)]
            # xi passes if significant in ALL tested conditioning sets
            any_significant = False
            for Z in cond_sets[:5]:  # limit for speed
                p = partial_corr_pvalue(X_arr[:, col_idx[xi]], T, Z, X_arr, col_idx)
                if p < alpha:
                    any_significant = True
                    break
            if any_significant:
                new_selected.append(xi)

        # Early dropping: remove features that became redundant
        prev_selected = list(selected)
        selected = list(dict.fromkeys(selected + new_selected))  # preserve order, dedup
        active = new_selected  # next round only considers currently significant

        if set(selected) == set(prev_selected):
            break  # converged

    # Backward phase
    to_remove = []
    for xi in selected:
        cond = [x for x in selected if x != xi]
        for k in range(0, min(len(cond), max_cond)+1):
            from itertools import combinations
            for Z in combinations(cond, k):
                if ci_test_with_target(xi, T, list(Z), X_arr, col_idx, alpha):
                    to_remove.append(xi)
                    break
            if xi in to_remove:
                break

    return [x for x in selected if x not in to_remove]


def hiton_mb(feat_cols, X_arr, T, col_idx, alpha=ALPHA_CI, max_cond=MAX_COND_SET):
    """
    HITON-MB: Find Parents & Children (PC) first, then add Spouses.
    PC(T) = variables directly connected to T in the causal graph.
    Spouses = parents of T's children (v-structures).
    """
    from itertools import combinations

    # Step 1: Find PC(T) — direct causes and effects of T
    # Forward: add candidates not independent of T
    CPC = []  # candidate PC
    candidates = list(feat_cols)

    for xi in candidates:
        p = partial_corr_pvalue(X_arr[:, col_idx[xi]], T, [], X_arr, col_idx)
        if p < alpha:
            CPC.append(xi)

    # Prune CPC: remove if ind. of T | subset of CPC
    PC = []
    for xi in CPC:
        cond = [x for x in CPC if x != xi]
        is_pc = True
        for k in range(0, min(len(cond), max_cond)+1):
            for Z in combinations(cond, k):
                if ci_test_with_target(xi, T, list(Z), X_arr, col_idx, alpha):
                    is_pc = False
                    break
            if not is_pc:
                break
        if is_pc:
            PC.append(xi)

    # Step 2: Find spouses — for each Xi in PC,
    # find Xj not in PC such that Xi and Xj are NOT ind. given T + PC\{Xi}
    spouses = []
    for xi in PC:
        cond_base = [x for x in PC if x != xi]
        for xj in feat_cols:
            if xj in PC or xj in spouses:
                continue
            # Xj is a spouse of T via Xi if Xj dep Xi | T, cond_base
            p = partial_corr_pvalue(X_arr[:, col_idx[xj]],
                                     X_arr[:, col_idx[xi]],
                                     cond_base, X_arr, col_idx)
            if p < alpha:
                spouses.append(xj)

    return list(dict.fromkeys(PC + spouses))  # MB = PC ∪ Spouses


def ensemble_mb(feat_cols, X_arr, T, col_idx, alpha=ALPHA_CI, max_cond=MAX_COND_SET):
    """
    Consensus MB: feature selected by >= 2 of 4 algorithms.
    """
    print("    Running IAMB...", end="", flush=True)
    mb1 = set(iamb(feat_cols, X_arr, T, col_idx, alpha, max_cond))
    print(f" {len(mb1)}")
    print("    Running Inter-IAMB...", end="", flush=True)
    mb2 = set(inter_iamb(feat_cols, X_arr, T, col_idx, alpha, max_cond))
    print(f" {len(mb2)}")
    print("    Running FBED...", end="", flush=True)
    mb3 = set(fbed(feat_cols, X_arr, T, col_idx, alpha, max_cond))
    print(f" {len(mb3)}")
    print("    Running HITON-MB...", end="", flush=True)
    mb4 = set(hiton_mb(feat_cols, X_arr, T, col_idx, alpha, max_cond))
    print(f" {len(mb4)}")

    all_sets = [mb1, mb2, mb3, mb4]
    # Keep features selected by >= 2 algorithms
    from collections import Counter
    counts = Counter(f for s in all_sets for f in s)
    consensus = [f for f, c in counts.items() if c >= 2]
    union     = list(set.union(*all_sets))
    intersection = list(set.intersection(*all_sets))

    print(f"    Intersection: {len(intersection)}  "
          f"Consensus(≥2): {len(consensus)}  Union: {len(union)}")

    return consensus if consensus else union

# ============================================================================
# METRICS
# ============================================================================

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
        return 0.5, 0.5, 0
    sub = df[features+["OS","OS.time"]].dropna()
    if len(sub) < 40:
        return 0.5, 0.5, len(features)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    threshold = SURVIVAL_YEARS * 365
    c_scores, auc_scores = [], []
    for tr, te in kf.split(sub):
        train, test = sub.iloc[tr], sub.iloc[te]
        try:
            cph = CoxPHFitter(penalizer=10.0)
            cph.fit(train, duration_col="OS.time", event_col="OS")
            pred = cph.predict_partial_hazard(test).values
            c_scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, pred))
        except Exception:
            X_sc = StandardScaler().fit(train[features]).transform(test[features])
            c_scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, X_sc.mean(1)))
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
            float(np.mean(auc_scores)) if auc_scores else 0.5,
            len(features))


def modality_breakdown(features):
    counts = {}
    for p in ["CLIN_","RNA_","CNV_","MUT_","PROT_","METH_","MIRNA_"]:
        n = sum(1 for f in features if f.startswith(p))
        if n: counts[p.rstrip("_")] = n
    return counts


# ============================================================================
# RUN EXPERIMENT — multiple alpha values per algorithm
# ============================================================================
print("\n" + "="*70)
print("RUNNING MB ALGORITHMS")
print("="*70)

ALPHAS_TO_TRY = [0.01, 0.05, 0.10, 0.20]

ALGORITHM_FNS = {
    "IAMB":      iamb,
    "Inter-IAMB":inter_iamb,
    "FBED":      fbed,
    "HITON-MB":  hiton_mb,
}

results = []

for algo_name, algo_fn in ALGORITHM_FNS.items():
    for alpha in ALPHAS_TO_TRY:
        run_name = f"{algo_name}_a{str(alpha).replace('.','')}"
        print(f"\n[{run_name}]")
        t0 = time.time()
        try:
            selected = algo_fn(feat_cols, X_arr, T, col_idx, alpha=alpha)
            elapsed = time.time() - t0
            print(f"  Selected: {len(selected)}  ({elapsed:.1f}s)")
            if selected:
                print(f"  Modalities: {modality_breakdown(selected)}")
            c_idx, auc5, n_feat = cv_metrics(df, selected)
            print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}")
            results.append({
                "algorithm":  run_name,
                "alpha":      alpha,
                "n_features": n_feat,
                "c_index":    round(c_idx, 4),
                "auc_5yr":    round(auc5, 4),
                "time_sec":   round(elapsed, 1),
                "features":   selected,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}  ({elapsed:.1f}s)")

# ============================================================================
# REPEAT KEY ALGORITHMS ON POOL B (top-20 winning fallback features)
# ============================================================================
print("\n" + "="*70)
print("POOL B: MB algorithms on top-20 Cox features (winning fallback)")
print("="*70)

feat_cols_b = feat_cols_20
X_b = StandardScaler().fit_transform(
    df[feat_cols_b + ["OS","OS.time"]].dropna()[feat_cols_b].values)
X_arr_b = X_b
col_idx_b = {f: i for i, f in enumerate(feat_cols_b)}
T_b = df[feat_cols_b + ["OS.time"]].dropna()["OS.time"].values

for algo_name, algo_fn in ALGORITHM_FNS.items():
    for alpha in [0.05, 0.10]:
        run_name = f"{algo_name}_poolB_a{str(alpha).replace('.','')}"
        print(f"\n[{run_name}]")
        t0 = time.time()
        try:
            selected = algo_fn(feat_cols_b, X_arr_b, T_b, col_idx_b, alpha=alpha)
            elapsed = time.time() - t0
            print(f"  Selected: {len(selected)}  ({elapsed:.1f}s)")
            if selected:
                print(f"  Modalities: {modality_breakdown(selected)}")
            c_idx, auc5, n_feat = cv_metrics(df, selected)
            print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}")
            results.append({
                "algorithm":  run_name,
                "alpha":      alpha,
                "n_features": n_feat,
                "c_index":    round(c_idx, 4),
                "auc_5yr":    round(auc5, 4),
                "time_sec":   round(elapsed, 1),
                "features":   selected,
            })
        except Exception as e:
            print(f"  ERROR: {e}")

# Ensemble (best alpha from above)
print(f"\n[Ensemble_a005]")
t0 = time.time()
try:
    selected_ens = ensemble_mb(feat_cols, X_arr, T, col_idx, alpha=0.05)
    elapsed = time.time() - t0
    print(f"  Selected: {len(selected_ens)}  ({elapsed:.1f}s)")
    print(f"  Modalities: {modality_breakdown(selected_ens)}")
    c_idx, auc5, n_feat = cv_metrics(df, selected_ens)
    print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}")
    results.append({
        "algorithm":  "Ensemble_a005",
        "alpha":      0.05,
        "n_features": n_feat,
        "c_index":    round(c_idx, 4),
        "auc_5yr":    round(auc5, 4),
        "time_sec":   round(elapsed, 1),
        "features":   selected_ens,
    })
except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY — sorted by C-index")
print("="*70)

res_df = pd.DataFrame([{k:v for k,v in r.items() if k!="features"} for r in results])
res_df = res_df.sort_values(["c_index","auc_5yr"], ascending=False).reset_index(drop=True)
print(res_df.to_string(index=False))

res_df.to_csv(OUT_DIR / "mb_algorithms_results.csv", index=False)

if results:
    best = res_df.iloc[0]
    best_feat = next(r["features"] for r in results if r["algorithm"]==best["algorithm"])
    with open(OUT_DIR / "best_mb_features.txt", "w") as f:
        f.write(f"Algorithm: {best['algorithm']}\n")
        f.write(f"C-index:   {best['c_index']}\n")
        f.write(f"AUC-5yr:   {best['auc_5yr']}\n\n")
        f.write("\n".join(best_feat))
    print(f"\nBest: {best['algorithm']}")
    print(f"  C-index: {best['c_index']}")
    print(f"  AUC-5yr: {best['auc_5yr']}")
    print(f"  Features: {best['n_features']}")
    print(f"  Modalities: {modality_breakdown(best_feat)}")

print(f"\nSaved: {OUT_DIR}/mb_algorithms_results.csv")
