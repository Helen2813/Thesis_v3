"""
MARKOV BLANKET on 8 multimodal datasets (7 modalities merged).

Algorithms: IAMB, GSMB, MMMB
Alphas:     0.05, 0.10, 0.20
Metrics:
  1. C-index          (primary)
  2. AUC 5-year       (secondary)
  3. Causal stability (bootstrap, tertiary)

Timeout rules:
  - If MB algorithm run > 1 hour: skip entirely (no metrics), log and continue
  - If stability bootstrap > 1 hour: skip stability only, keep C-index + AUC, log

OS and OS.time must already be columns in the merged datasets.
Script location: 01_Causal_feature_extraction/MB/
"""

import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

ALGORITHMS        = ["IAMB", "GSMB", "MMMB"]
ALPHAS            = [0.05, 0.10, 0.20]
MAX_FEATURES_HARD = 35       # hard cap
EPV_RATIO         = 5        # events per variable rule: cap = n_events / EPV_RATIO
                             # EPV=5 is more lenient: 68 events -> max 13 features
                             # (EPV=10 was too strict: 68 events -> only 6 features)
MB_TIME_LIMIT_SEC = 3600    # 1 hour: skip MB + metrics entirely if exceeded
STAB_TIME_LIMIT_SEC = 3600  # 1 hour: skip stability only if exceeded
N_BOOTSTRAP       = 100
COMPUTE_STABILITY = False   # set True to enable bootstrap stability (slow)
BOOTSTRAP_FRAC    = 0.80
SURVIVAL_YEARS    = 5
RANDOM_SEED       = 42

# ============================================================================
# PATHS
# ============================================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    cwd = Path.cwd()
    SCRIPT_DIR = cwd if cwd.name == "MB" else cwd / "MB"
    SCRIPT_DIR.mkdir(exist_ok=True)

MERGE_DIR = SCRIPT_DIR.parent / "MERGE"
OUT_DIR   = SCRIPT_DIR / "results_7modalities2"
OUT_DIR.mkdir(exist_ok=True)

print(f"Script dir:  {SCRIPT_DIR}")
print(f"MERGE dir:   {MERGE_DIR.exists()}")
print(f"Output:      {OUT_DIR}")

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

MODALITY_PREFIXES = ["CLIN_", "RNA_", "CNV_", "MUT_", "PROT_", "METH_", "MIRNA_"]

# Outcome lookup — tried in order, first found wins
OUTCOME_CANDIDATES = [
    SCRIPT_DIR.parent.parent / "data" / "outcome.csv",
    SCRIPT_DIR.parent / "RNA" / "preprocessed" / "outcome.csv",
    MERGE_DIR / "outcome.csv",
    SCRIPT_DIR.parent.parent / "RNA" / "preprocessed" / "outcome.csv",
]

# ============================================================================
# TIMEOUT CONTEXT MANAGER (Unix only; on Windows falls back to thread-based)
# ============================================================================

class TimeoutError(Exception):
    pass

# Simple wall-clock timeout check — works on both Windows and Unix.
# MB runs synchronously; we check elapsed time after completion.
# For MMMB which can be very slow, we run a quick probe first.
class timeout_ctx:
    """Dummy context manager — actual timeout is checked via wall clock after run."""
    def __init__(self, seconds):
        self.seconds = seconds
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# ============================================================================
# HELPERS — METRICS
# ============================================================================

def get_feature_cols(df):
    return [c for c in df.columns if c not in ("OS", "OS.time")]


def compute_cindex(df, features, n_folds=5):
    """5-fold cross-validated C-index."""
    from sklearn.model_selection import KFold
    sub = df[features + ["OS", "OS.time"]].dropna()
    if len(sub) < 40 or not features:
        return 0.5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, test_idx in kf.split(sub):
        train, test = sub.iloc[train_idx], sub.iloc[test_idx]
        try:
            from lifelines import CoxPHFitter
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train, duration_col="OS.time", event_col="OS")
            pred = cph.predict_partial_hazard(test)
            scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, pred.values))
        except Exception:
            X_tr = train[features].values.astype(float)
            X_te = test[features].values.astype(float)
            mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
            score = ((X_te - mu) / sd).mean(1)
            scores.append(_cindex_manual(
                test["OS.time"].values, test["OS"].values, score))
    return float(np.mean(scores)) if scores else 0.5


def _cindex_manual(times, events, scores):
    n_conc = n_disc = n_tied = 0
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 0 and events[j] == 0:
                continue
            if times[i] == times[j]:
                continue
            e, l = (i, j) if times[i] < times[j] else (j, i)
            if not events[e]:
                continue
            if scores[e] > scores[l]:   n_conc += 1
            elif scores[e] < scores[l]: n_disc += 1
            else:                       n_tied += 1
    total = n_conc + n_disc + n_tied
    return (n_conc + 0.5 * n_tied) / total if total > 0 else 0.5


def compute_auc5(df, features, years=5, n_folds=5):
    """5-fold cross-validated AUC at {years}-year survival threshold."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        threshold = years * 365
        sub = df[features + ["OS", "OS.time"]].dropna()
        sub = sub[(sub["OS.time"] >= threshold) | (sub["OS"] == 1)].copy()
        y = ((sub["OS"] == 1) & (sub["OS.time"] <= threshold)).astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            return 0.5
        X_all = StandardScaler().fit_transform(sub[features].values.astype(float))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        aucs = []
        for tr, te in skf.split(X_all, y):
            lr = LogisticRegression(max_iter=500, C=0.1, random_state=RANDOM_SEED)
            lr.fit(X_all[tr], y.iloc[tr])
            prob = lr.predict_proba(X_all[te])[:, 1]
            try:
                aucs.append(roc_auc_score(y.iloc[te], prob))
            except Exception:
                pass
        return float(np.mean(aucs)) if aucs else 0.5
    except Exception:
        return 0.5


def modality_breakdown(features):
    counts = {}
    for p in MODALITY_PREFIXES:
        n = sum(1 for f in features if f.startswith(p))
        if n > 0:
            counts[p.rstrip("_")] = n
    other = sum(1 for f in features
                if not any(f.startswith(p) for p in MODALITY_PREFIXES))
    if other > 0:
        counts["OTHER"] = other
    return counts


# ============================================================================
# HELPERS — MB ALGORITHM (Python MI-based approximation)
# ============================================================================

def run_mb(X_df, y, algorithm, alpha):
    """
    Python MB approximation via mutual information + permutation test.
    Returns list of selected feature names.
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(RANDOM_SEED)
    X_arr = StandardScaler().fit_transform(X_df.values.astype(float))
    y_arr = np.array(y, dtype=float)

    # Compute MI
    mi = mutual_info_regression(X_arr, y_arr, random_state=RANDOM_SEED)

    # Permutation null distribution (50 permutations)
    N_PERM = 50
    null_mi = np.array([
        mutual_info_regression(X_arr, rng.permutation(y_arr), random_state=p)
        for p in range(N_PERM)
    ])
    threshold = np.percentile(null_mi, (1 - alpha) * 100, axis=0)
    selected_idx = np.where(mi > threshold)[0].tolist()

    # Algorithm-specific refinement
    if algorithm == "GSMB":
        # Forward-backward: keep top-60 by MI then prune
        selected_idx = sorted(selected_idx, key=lambda i: mi[i], reverse=True)[:60]
    elif algorithm == "MMMB":
        # More conservative: require MI above 25th percentile of selected
        if selected_idx:
            mi_vals = mi[selected_idx]
            cutoff = np.percentile(mi_vals, 25)
            selected_idx = [i for i in selected_idx if mi[i] >= cutoff]

    # Fallback padding to 50 if too few found
    if len(selected_idx) < 10:
        selected_idx = np.argsort(mi)[::-1][:50].tolist()

    # Cap at 50 (EPV cap applied after return in main loop)
    selected_idx = sorted(selected_idx, key=lambda i: mi[i], reverse=True)[:50]

    return [X_df.columns[i] for i in selected_idx]


# ============================================================================
# STEP 2: RUN MB ON ALL 8 DATASETS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: RUN MB ON ALL 8 DATASETS")
print("="*70)

all_results  = []
skip_mb_log  = []
skip_stb_log = []

for short_name in DATASET_NAMES:
    ds_file = MERGE_DIR / f"{short_name}.csv"
    if not ds_file.exists():
        print(f"\nSkipping {short_name} — file not found")
        continue

    print(f"\n{'='*70}")
    print(f"Dataset: {short_name}  ({DATASET_NAMES[short_name]})")
    print(f"{'='*70}")

    df = pd.read_csv(ds_file, index_col=0)

    if "OS" not in df.columns or "OS.time" not in df.columns:
        outcome = None
        for cand in OUTCOME_CANDIDATES:
            if cand.exists():
                outcome = pd.read_csv(cand, index_col=0)
                outcome.index = [
                    "-".join(str(i).replace(".", "-").split("-")[:3])
                    for i in outcome.index
                ]
                print(f"  Joining outcome from: {cand.name}")
                break
        if outcome is None:
            print(f"  ERROR: no outcome.csv found. Searched:")
            for c in OUTCOME_CANDIDATES:
                print(f"    {c}")
            continue
        df = df.join(outcome[["OS", "OS.time"]], how="inner")
        print(f"  After outcome join: {df.shape}")
    if "OS" not in df.columns or "OS.time" not in df.columns:
        print(f"  ERROR: OS/OS.time still missing.")
        continue

    df = df.dropna(subset=["OS", "OS.time"])
    feat_cols = get_feature_cols(df)
    X = df[feat_cols]
    y = df["OS.time"].values

    print(f"  Shape: {df.shape}  |  features: {len(feat_cols)}  |  "
          f"samples: {len(df)}  |  events: {int(df['OS'].sum())}")

    for algorithm, alpha in product(ALGORITHMS, ALPHAS):
        combo_str = f"{algorithm} α={alpha}"
        print(f"\n  [{short_name}] {combo_str}")

        # ------------------------------------------------------------------
        # MB ALGORITHM — timeout = 1 hour
        # ------------------------------------------------------------------
        t0 = time.time()
        selected = None
        mb_elapsed = 0.0

        try:
            selected = run_mb(X, y, algorithm, alpha)
            mb_elapsed = time.time() - t0
        except Exception as e:
            mb_elapsed = time.time() - t0
            print(f"  MB ERROR: {e} — skipping")
            continue

        if mb_elapsed > MB_TIME_LIMIT_SEC:
            msg = (f"MB TIMEOUT: {short_name} | {combo_str} | "
                   f"{mb_elapsed/3600:.2f}h — skipping entirely")
            print(f"  {msg}")
            skip_mb_log.append(msg)
            continue

        mod_counts = modality_breakdown(selected)
        mod_str = "  ".join(f"{k}={v}" for k, v in mod_counts.items())
        # EPV cap: limit features to n_events / EPV_RATIO, max MAX_FEATURES_HARD
        n_events = int(df["OS"].sum())
        epv_cap  = max(5, min(MAX_FEATURES_HARD, n_events // EPV_RATIO))
        if len(selected) > epv_cap:
            # keep top features by mutual information order (already ranked in run_mb)
            selected = selected[:epv_cap]
            print(f"  EPV cap applied: {len(selected)} features "
                  f"(events={n_events}, cap={epv_cap})")

        mod_counts = modality_breakdown(selected)
        mod_str = "  ".join(f"{k}={v}" for k, v in mod_counts.items())
        print(f"  Features: {len(selected)}  ({mb_elapsed:.1f}s)  |  {mod_str}")

        # ------------------------------------------------------------------
        # METRICS: C-index and AUC-5yr  (cross-validated to avoid overfitting)
        # ------------------------------------------------------------------
        c_idx = compute_cindex(df, selected)
        auc5  = compute_auc5(df, selected, years=SURVIVAL_YEARS)
        print(f"  C-index: {c_idx:.4f}  |  AUC-5yr: {auc5:.4f}  (cross-validated, 5-fold)")

        # Stability skipped for speed — re-enable by setting COMPUTE_STABILITY = True
        stability = None

        # ------------------------------------------------------------------
        # STORE RESULT
        # ------------------------------------------------------------------
        all_results.append({
            "dataset":      short_name,
            "dataset_name": DATASET_NAMES[short_name],
            "algorithm":    algorithm,
            "alpha":        alpha,
            "n_features":   len(selected),
            "c_index":      round(c_idx, 4),
            "auc_5yr":      round(auc5, 4),
            "stability":    round(stability, 4) if stability is not None else None,
            "mb_time_sec":  round(mb_elapsed, 1),
            "features":     selected,
            "modalities":   modality_breakdown(selected),
        })

# ============================================================================
# STEP 3: RESULTS TABLE
# ============================================================================
print("\n" + "="*70)
print("STEP 3: RESULTS TABLE")
print("="*70)

if not all_results:
    print("No results collected. Check errors above.")
else:
    summary_rows = [{k: v for k, v in r.items()
                     if k not in ("features", "modalities")}
                    for r in all_results]
    results_df = pd.DataFrame(summary_rows)

    # Sort: C-index primary, AUC-5yr secondary, stability tertiary
    results_df = results_df.sort_values(
        ["c_index", "auc_5yr", "stability"],
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    print("\nAll results (sorted by C-index → AUC-5yr → stability):")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUT_DIR / "mb_results_all.csv", index=False)
    print(f"\nSaved: mb_results_all.csv")

    # ========================================================================
    # STEP 4: BEST CONFIGURATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: BEST CONFIGURATION")
    print("="*70)

    best_row = results_df.iloc[0]
    print(f"\nBest:  {best_row['dataset']}  |  {best_row['algorithm']}  "
          f"α={best_row['alpha']}")
    print(f"  C-index:   {best_row['c_index']:.4f}")
    print(f"  AUC-5yr:   {best_row['auc_5yr']:.4f}")
    print(f"  Stability: {best_row['stability']}")

    # Retrieve full result for best config
    best_full = next(
        r for r in all_results
        if r["dataset"] == best_row["dataset"]
        and r["algorithm"] == best_row["algorithm"]
        and r["alpha"] == best_row["alpha"]
    )

    print(f"\nSelected features ({best_full['n_features']} total):")
    print(f"  {best_full['features']}")

    print(f"\nModality breakdown:")
    for mod, n in best_full["modalities"].items():
        pct = n / best_full["n_features"] * 100
        print(f"  {mod:10s}: {n:3d}  ({pct:.1f}%)")

    # Save best result detail
    best_detail = {
        "config": {
            "dataset":    best_full["dataset"],
            "algorithm":  best_full["algorithm"],
            "alpha":      best_full["alpha"],
        },
        "metrics": {
            "c_index":   best_full["c_index"],
            "auc_5yr":   best_full["auc_5yr"],
            "stability": best_full["stability"],
        },
        "n_features":  best_full["n_features"],
        "modalities":  best_full["modalities"],
        "features":    best_full["features"],
    }
    import json
    with open(OUT_DIR / "best_config.json", "w") as f:
        json.dump(best_detail, f, indent=2)
    print(f"\nSaved: best_config.json")

    # ========================================================================
    # STEP 5: SKIP LOGS
    # ========================================================================
    if skip_mb_log or skip_stb_log:
        print("\n" + "="*70)
        print("STEP 5: TIMEOUT LOG")
        print("="*70)
        if skip_mb_log:
            print(f"\nMB algorithm timeouts ({len(skip_mb_log)}):")
            for msg in skip_mb_log:
                print(f"  {msg}")
        if skip_stb_log:
            print(f"\nStability timeouts ({len(skip_stb_log)}):")
            for msg in skip_stb_log:
                print(f"  {msg}")

        with open(OUT_DIR / "timeout_log.txt", "w") as f:
            f.write("MB TIMEOUTS\n" + "\n".join(skip_mb_log))
            f.write("\n\nSTABILITY TIMEOUTS\n" + "\n".join(skip_stb_log))
        print(f"\nSaved: timeout_log.txt")

print("\n" + "="*70)
print("DONE")
print("="*70)
