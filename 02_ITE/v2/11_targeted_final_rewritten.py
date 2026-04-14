
"""
Targeted Therapy — Small-Sample Exploratory Analysis (rewritten)
================================================================
Clinical question:
Does adding HER2-targeted therapy on top of chemotherapy improve survival
among HER2-positive breast cancer patients?

Primary comparison:
    HER2+ patients who received chemotherapy:
        treated  = chemo + targeted
        control  = chemo only

Exploratory only:
    The treated sample is small (~25), so this script emphasizes
    agreement across multiple estimators and transparent diagnostics,
    not strong causal claims.

Key fixes vs prior version:
1. Correct primary control group: HER2+ chemo patients only.
2. True seed variation for AIPW and Simple IPW.
3. No silent swallowing of model failures — all failures are logged.
4. Uses discrete treatment for DML / CausalForestDML.
5. Reports overlap diagnostics (PS range, max stabilized IPW, ESS, mean SMD).
6. Clearly marks targeted therapy as exploratory.

Outputs:
    targeted_final_results_rewritten.csv
    targeted_final_summary_rewritten.csv
    targeted_final_figure_rewritten.png
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from econml.dml import LinearDML, CausalForestDML
from econml.dr import LinearDRLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
SEEDS = [42, 123, 456, 789, 1337, 2024, 31, 99, 7, 404]
CV_FOLDS = 3
MIN_N = 8

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]
     if (p / "02_ITE" / "01_preprocessing" / "output" / "ite_ready_dataset_v2.csv").exists()),
    None
)
if BASE_DIR is None:
    print("ERROR: ite_ready_dataset_v2.csv not found")
    sys.exit(1)

INPUT_DIR = BASE_DIR / "02_ITE" / "01_preprocessing" / "output"
OUTPUT_DIR = SCRIPT_DIR / "targeted_final_output_rewritten"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  TARGETED THERAPY — SMALL SAMPLE EXPLORATORY ANALYSIS")
print("=" * 72)

df = pd.read_csv(INPUT_DIR / "ite_ready_dataset_v2.csv")

drop_cols = [
    "T", "Y", "patient_id", "propensity_score",
    "T_hormone", "T_chemo", "T_targeted", "T_radiation", "T_hormone_excl"
]
X_df = df.drop(columns=drop_cols, errors="ignore")
X_all = X_df.values.astype(float)
FEAT = X_df.columns.tolist()
Y_all = df["Y"].astype(int).values

er = (df["ER_status"].fillna(0) > 0.5).astype(int).values
pr = (df["PR_status"].fillna(0) > 0.5).astype(int).values
her2 = (df["HER2_status"].fillna(0) > 0.5).astype(int).values

T_targeted = df["T_targeted"].astype(int).values
T_chemo = df["T_chemo"].astype(int).values

# Primary and auxiliary masks
her2pos = (her2 == 1)
her2pos_chemo = her2pos & (T_chemo == 1)           # correct clinical comparison
her2neg_chemo = (her2 == 0) & (T_chemo == 1)       # attempted falsification

print("\n  CONTROL GROUP: HER2+ chemo patients only")
print("  Clinical question: chemo+targeted vs chemo alone in HER2+\n")

for name, mask, T_vec in [
    ("HER2+ chemo patients (primary)", her2pos_chemo, T_targeted),
    ("All HER2+ (old approach)", her2pos, T_targeted),
    ("HER2- chemo patients (falsif)", her2neg_chemo, T_targeted),
    ("All patients", np.ones(len(df), dtype=bool), T_targeted),
]:
    m = mask.astype(bool)
    t = int(T_vec[m].sum())
    c = int((T_vec[m] == 0).sum())
    print(f"    {name:<40} n={m.sum():4d}  t={t:3d}  c={c:3d}")

print()


# -----------------------------------------------------------------------------
# Helper models
# -----------------------------------------------------------------------------
def rf_reg(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )


def rf_clf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def ridge_reg(seed: int) -> RidgeCV:
    return RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def propensity_scores(X: np.ndarray, T: np.ndarray, seed: int = RANDOM_STATE) -> np.ndarray:
    lr = LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced")
    lr.fit(X, T)
    return lr.predict_proba(X)[:, 1]


def stabilized_ipw(X: np.ndarray, T: np.ndarray, clip=(0.05, 0.95), seed: int = RANDOM_STATE) -> np.ndarray:
    ps = propensity_scores(X, T, seed=seed).clip(*clip)
    p1 = T.mean()
    sw = np.where(T == 1, p1 / ps, (1 - p1) / (1 - ps))
    return sw / sw.mean()


def ess(weights: np.ndarray) -> float:
    return (weights.sum() ** 2) / np.sum(weights ** 2)


def overlap_diagnostics(X: np.ndarray, T: np.ndarray, feat_cap: int = 10) -> dict:
    ps = propensity_scores(X, T)
    sw = stabilized_ipw(X, T)
    smds = []
    for i in range(min(X.shape[1], feat_cap)):
        x = X[:, i]
        xt = x[T == 1]
        xc = x[T == 0]
        pooled = np.sqrt((xt.std() ** 2 + xc.std() ** 2) / 2) + 1e-8
        smd = abs((xt.mean() - xc.mean()) / pooled)
        smds.append(float(smd))
    return {
        "ps_min": float(ps.min()),
        "ps_max": float(ps.max()),
        "max_ipw": float(sw.max()),
        "ess": float(ess(sw)),
        "mean_smd": float(np.mean(smds)) if smds else np.nan,
    }


def print_diag(diag: dict, n: int) -> None:
    print(
        f"    PS: [{diag['ps_min']:.3f},{diag['ps_max']:.3f}]  "
        f"maxIPW={diag['max_ipw']:.1f}  ESS={int(round(diag['ess']))}/{n}  "
        f"SMD={diag['mean_smd']:.3f}"
    )


# -----------------------------------------------------------------------------
# Transparent estimators
# -----------------------------------------------------------------------------
def simple_ipw_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int, clip=(0.05, 0.95)) -> float:
    ps = propensity_scores(X, T, seed=seed).clip(*clip)
    return float(np.mean(T * Y / ps - (1 - T) * Y / (1 - ps)))


def aipw_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int, n_folds: int = 3) -> float:
    """
    Cross-fitted AIPW for binary outcome and binary treatment.
    Uses seed-sensitive CV splits and nuisance fits.
    """
    n = len(Y)
    cv = min(n_folds, int(T.sum()), int((T == 0).sum()))
    if cv < 2:
        raise ValueError("Not enough treated/control samples for AIPW CV.")

    mu1 = np.zeros(n, dtype=float)
    mu0 = np.zeros(n, dtype=float)
    ps = np.zeros(n, dtype=float)

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    for tr_idx, val_idx in kf.split(X, T):
        Xtr, Xval = X[tr_idx], X[val_idx]
        Ttr, Ytr = T[tr_idx], Y[tr_idx]

        # Outcome models for treated and control
        y_model_t = LogisticRegression(
            max_iter=500, random_state=seed, class_weight="balanced"
        )
        y_model_c = LogisticRegression(
            max_iter=500, random_state=seed + 1, class_weight="balanced"
        )

        Xtr_t = Xtr[Ttr == 1]
        Ytr_t = Ytr[Ttr == 1]
        Xtr_c = Xtr[Ttr == 0]
        Ytr_c = Ytr[Ttr == 0]

        if len(Xtr_t) > 1 and len(np.unique(Ytr_t)) > 1:
            y_model_t.fit(Xtr_t, Ytr_t)
            mu1[val_idx] = y_model_t.predict_proba(Xval)[:, 1]
        else:
            mu1[val_idx] = float(np.mean(Ytr_t)) if len(Ytr_t) > 0 else 0.0

        if len(Xtr_c) > 1 and len(np.unique(Ytr_c)) > 1:
            y_model_c.fit(Xtr_c, Ytr_c)
            mu0[val_idx] = y_model_c.predict_proba(Xval)[:, 1]
        else:
            mu0[val_idx] = float(np.mean(Ytr_c)) if len(Ytr_c) > 0 else 0.0

        # Propensity model
        t_model = LogisticRegression(
            max_iter=500, random_state=seed + 2, class_weight="balanced"
        )
        t_model.fit(Xtr, Ttr)
        ps[val_idx] = t_model.predict_proba(Xval)[:, 1].clip(0.05, 0.95)

    score = mu1 - mu0 + T * (Y - mu1) / ps - (1 - T) * (Y - mu0) / (1 - ps)
    return float(np.mean(score))


# -----------------------------------------------------------------------------
# Repeated-seed econml runners
# -----------------------------------------------------------------------------
def summarize_repeats(values):
    arr = np.asarray(values, dtype=float)
    pct_prot = float(np.mean(arr < 0) * 100)
    return {
        "median_ATE": float(np.median(arr)),
        "ATE_min": float(arr.min()),
        "ATE_max": float(arr.max()),
        "IQR_lo": float(np.percentile(arr, 25)),
        "IQR_hi": float(np.percentile(arr, 75)),
        "pct_protective": pct_prot,
        "stable": bool(pct_prot >= 80 or pct_prot <= 20),
        "direction": "protective" if np.median(arr) < 0 else "harmful",
    }


def run_econml_repeated(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    label: str,
    technique: str,
    model_y_factory,
    model_t_factory,
    sample_weight=None,
):
    n1 = int(T.sum())
    n0 = int((T == 0).sum())
    if n1 < MIN_N or n0 < MIN_N:
        print(f"    [{technique}] SKIP: n1={n1}, n0={n0}")
        return []

    lin_vals, cf_vals, dr_vals = [], [], []
    failures = []

    for seed in SEEDS:
        cv = min(CV_FOLDS, n1, n0)

        # LinearDML
        try:
            lin = LinearDML(
                model_y=model_y_factory(seed),
                model_t=model_t_factory(seed),
                discrete_treatment=True,
                linear_first_stages=False,
                cv=cv,
                random_state=seed,
            )
            kwargs = {"inference": "statsmodels"}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
            lin.fit(Y.astype(float), T, X=X.astype(float), **kwargs)
            lin_vals.append(float(np.mean(lin.effect(X.astype(float)))))
        except Exception as e:
            failures.append(f"LinearDML(seed={seed}): {e}")

        # CausalForestDML
        try:
            cf = CausalForestDML(
                model_y=model_y_factory(seed),
                model_t=model_t_factory(seed),
                discrete_treatment=True,
                n_estimators=200,
                min_samples_leaf=5,
                max_features="sqrt",
                cv=cv,
                random_state=seed,
                n_jobs=-1,
            )
            kwargs = {}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
            cf.fit(Y.astype(float), T, X=X.astype(float), **kwargs)
            cf_vals.append(float(np.mean(cf.effect(X.astype(float)))))
        except Exception as e:
            failures.append(f"CausalForestDML(seed={seed}): {e}")

        # LinearDRLearner
        try:
            dr = LinearDRLearner(
                model_regression=model_y_factory(seed),
                model_propensity=model_t_factory(seed),
                cv=cv,
                random_state=seed,
            )
            kwargs = {}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
            dr.fit(Y.astype(float), T, X=X.astype(float), **kwargs)
            dr_vals.append(float(np.mean(dr.effect(X.astype(float)))))
        except Exception as e:
            failures.append(f"LinearDRLearner(seed={seed}): {e}")

    results = []
    for vals, model_name in [
        (lin_vals, "LinearDML"),
        (cf_vals, "CausalForestDML"),
        (dr_vals, "LinearDRLearner"),
    ]:
        if not vals:
            print(f"    [{technique}] {model_name:<20} FAILED on all seeds")
            continue
        s = summarize_repeats(vals)
        print(
            f"    [{technique}] {model_name:<20} median={s['median_ATE']:+.4f}  "
            f"range=[{s['ATE_min']:+.4f},{s['ATE_max']:+.4f}]  "
            f"protective={s['pct_protective']:.0f}%  {'✓' if s['stable'] else '~'}"
        )
        results.append({
            "label": label,
            "technique": technique,
            "model": model_name,
            "n_treated": n1,
            "n_control": n0,
            **s,
        })

    if failures:
        fail_path = OUTPUT_DIR / f"{label.lower()}_{technique}_failures.log"
        fail_path.write_text("\n".join(failures), encoding="utf-8")

    return results


# -----------------------------------------------------------------------------
# Run primary targeted analysis
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  PRIMARY: HER2+ chemo patients × T_targeted")
print("  (chemo+targeted vs chemo alone — clinically appropriate comparison)")
print("=" * 72)

mask_primary = her2pos_chemo.astype(bool)
X = X_all[mask_primary]
T = T_targeted[mask_primary].astype(int)
Y = Y_all[mask_primary].astype(int)
n1 = int(T.sum())
n0 = int((T == 0).sum())

print(f"\n  n={mask_primary.sum()}, treated={n1}, control={n0}")
diag_primary = overlap_diagnostics(X, T)
print_diag(diag_primary, len(T))

all_results = []

# T1 — baseline
print("\n  T1. RF baseline (DML / CF / DR)")
all_results.extend(run_econml_repeated(X, T, Y, "PRIMARY", "T1_rf_baseline", rf_reg, rf_clf))

# T2 — stabilized IPW
print("\n  T2. + Stabilized IPW")
sw = stabilized_ipw(X, T)
all_results.extend(run_econml_repeated(X, T, Y, "PRIMARY", "T2_sipw", rf_reg, rf_clf, sample_weight=sw))

# T3 — PS trim
ps_vals = propensity_scores(X, T)
keep = (ps_vals >= 0.10) & (ps_vals <= 0.90)
print(f"\n  T3. PS trimming [0.10,0.90]  kept={int(keep.sum())}/{len(keep)}")
if keep.sum() > 15 and int(T[keep].sum()) >= MIN_N and int((T[keep] == 0).sum()) >= MIN_N:
    all_results.extend(run_econml_repeated(X[keep], T[keep], Y[keep], "PRIMARY", "T3_ps_trim", rf_reg, rf_clf))
else:
    print("    [T3_ps_trim] SKIP: insufficient treated/control after trim")

# T4 — ridge outcome
print("\n  T4. Ridge penalized outcome model")
all_results.extend(run_econml_repeated(X, T, Y, "PRIMARY", "T4_ridge_outcome", ridge_reg, rf_clf))

# T6 — AIPW with true seed variation
print("\n  T6. AIPW (cross-fitted, seed-varying)")
aipw_vals = []
for seed in SEEDS:
    try:
        aipw_vals.append(aipw_ate(X, T, Y, seed=seed))
    except Exception as e:
        print(f"    [T6_aipw] seed={seed} failed: {e}")
if aipw_vals:
    s = summarize_repeats(aipw_vals)
    print(
        f"    median={s['median_ATE']:+.4f}  "
        f"range=[{s['ATE_min']:+.4f},{s['ATE_max']:+.4f}]  "
        f"protective={s['pct_protective']:.0f}%  {'✓' if s['stable'] else '~'}"
    )
    all_results.append({
        "label": "PRIMARY",
        "technique": "T6_aipw",
        "model": "AIPW",
        "n_treated": n1,
        "n_control": n0,
        **s,
    })

# T7 — simple IPW with true seed variation
print("\n  T7. Simple IPW (transparent non-parametric baseline)")
simple_vals = []
for seed in SEEDS:
    try:
        simple_vals.append(simple_ipw_ate(X, T, Y, seed=seed))
    except Exception as e:
        print(f"    [T7_simple_ipw] seed={seed} failed: {e}")
if simple_vals:
    s = summarize_repeats(simple_vals)
    print(
        f"    median={s['median_ATE']:+.4f}  "
        f"range=[{s['ATE_min']:+.4f},{s['ATE_max']:+.4f}]  "
        f"protective={s['pct_protective']:.0f}%  {'✓' if s['stable'] else '~'}"
    )
    all_results.append({
        "label": "PRIMARY",
        "technique": "T7_simple_ipw",
        "model": "SimpleIPW",
        "n_treated": n1,
        "n_control": n0,
        **s,
    })


# -----------------------------------------------------------------------------
# Attempted falsification
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  FALSIFICATION: HER2- chemo patients × T_targeted")
print("  (anti-HER2 in HER2-negative — should be near-zero/harmful if estimable)")
print("=" * 72)

mask_f = her2neg_chemo.astype(bool)
Xf = X_all[mask_f]
Tf = T_targeted[mask_f].astype(int)
Yf = Y_all[mask_f].astype(int)
nf1 = int(Tf.sum())
nf0 = int((Tf == 0).sum())

print(f"\n  n={mask_f.sum()}, treated={nf1}, control={nf0}")
if nf1 >= MIN_N and nf0 >= MIN_N:
    print_diag(overlap_diagnostics(Xf, Tf), len(Tf))
    all_results.extend(run_econml_repeated(Xf, Tf, Yf, "FALSIF", "T1_baseline", rf_reg, rf_clf))
else:
    print(f"    SKIP FALSIF: n1={nf1}, n0={nf0} — too few treated cases for estimation")


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  RESULTS SUMMARY  (⚠ EXPLORATORY — targeted n_treated is very small)")
print("=" * 72)

res_df = pd.DataFrame(all_results)
res_path = OUTPUT_DIR / "targeted_final_results_rewritten.csv"
res_df.to_csv(res_path, index=False)

primary = res_df[res_df["label"] == "PRIMARY"].copy()
falsif = res_df[res_df["label"] == "FALSIF"].copy()

if not primary.empty:
    print("\n  PRIMARY — technique comparison:")
    print(primary[[
        "technique", "model", "median_ATE", "ATE_min", "ATE_max",
        "pct_protective", "stable", "direction"
    ]].to_string(index=False))

    prot_frac = float(np.mean(primary["median_ATE"] < 0))
    print(f"\n  Technique agreement: {prot_frac * 100:.0f}% show protective direction")
    print(f"  Number of technique/model rows: {len(primary)}")
    print(f"  Sign distribution: {primary['direction'].value_counts().to_dict()}")

if not falsif.empty:
    print("\n  FALSIFICATION:")
    print(falsif[[
        "technique", "model", "median_ATE", "pct_protective", "stable", "direction"
    ]].to_string(index=False))
else:
    print("\n  FALSIFICATION:")
    print("  Not estimable — too few HER2- targeted cases.")

summary_rows = []
if not primary.empty:
    median_primary = float(primary["median_ATE"].median())
    min_primary = float(primary["ATE_min"].min())
    max_primary = float(primary["ATE_max"].max())
    summary_rows.append({
        "section": "primary",
        "n_rows": len(primary),
        "median_of_medians": median_primary,
        "global_min": min_primary,
        "global_max": max_primary,
        "protective_fraction": float(np.mean(primary["median_ATE"] < 0)),
        "comment": "Exploratory targeted-therapy result using HER2+ chemo-treated comparison group",
    })
if not falsif.empty:
    summary_rows.append({
        "section": "falsification",
        "n_rows": len(falsif),
        "median_of_medians": float(falsif["median_ATE"].median()),
        "global_min": float(falsif["ATE_min"].min()),
        "global_max": float(falsif["ATE_max"].max()),
        "protective_fraction": float(np.mean(falsif["median_ATE"] < 0)),
        "comment": "Only if enough HER2- targeted cases exist",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "targeted_final_summary_rewritten.csv", index=False)

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
if not primary.empty:
    sub = primary.copy().reset_index(drop=True)
    sub["y_label"] = sub["technique"] + " | " + sub["model"]
    colors = [
        "#2ca02c" if v < -0.003 else "#ff7f0e" if abs(v) <= 0.003 else "#d62728"
        for v in sub["median_ATE"]
    ]
    ax.barh(range(len(sub)), sub["median_ATE"], color=colors, alpha=0.85)
    ax.errorbar(
        sub["median_ATE"],
        range(len(sub)),
        xerr=[sub["median_ATE"] - sub["IQR_lo"], sub["IQR_hi"] - sub["median_ATE"]],
        fmt="none", color="black", capsize=3, lw=1.2
    )
    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["y_label"], fontsize=7)
    ax.set_xlabel("Median ATE (10 seeds)")
    ax.set_title(
        "PRIMARY: HER2+ chemo × T_targeted\n(chemo+targeted vs chemo alone)",
        fontweight="bold", fontsize=9
    )

ax2 = axes[1]
if not falsif.empty:
    subf = falsif.copy().reset_index(drop=True)
    colors_f = [
        "#2ca02c" if v < -0.003 else "#ff7f0e" if abs(v) <= 0.003 else "#d62728"
        for v in subf["median_ATE"]
    ]
    ax2.barh(range(len(subf)), subf["median_ATE"], color=colors_f, alpha=0.85)
    ax2.axvline(0, color="gray", linestyle="--", lw=1)
    ax2.set_yticks(range(len(subf)))
    ax2.set_yticklabels(subf["technique"] + " | " + subf["model"], fontsize=8)
    ax2.set_xlabel("Median ATE")
    ax2.set_title(
        "FALSIFICATION: HER2- chemo × T_targeted",
        fontweight="bold", fontsize=9
    )
else:
    ax2.text(
        0.5, 0.5,
        "No usable falsification cohort\n(too few HER2- targeted cases)",
        ha="center", va="center", fontsize=10
    )
    ax2.axis("off")

plt.suptitle(
    "Targeted Therapy — Exploratory Small-Sample Analysis\n"
    "⚠ Interpret cautiously: treated sample is very small",
    fontsize=10
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "targeted_final_figure_rewritten.png", bbox_inches="tight", dpi=150)
plt.close(fig)

print(f"\n  Saved: {OUTPUT_DIR}")
print("=" * 72)
print("  DONE")
print("=" * 72)
