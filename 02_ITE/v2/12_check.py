"""
Final ITE evaluation script for thesis reporting.

Outputs:
- final_ite_results_long.csv
- final_ite_results_wide.csv
- final_ite_seedwise_ate.csv
- final_ite_placebo.csv
- final_ite_overlap.csv
- final_ite_policy.csv
- final_ite_summary.json
- fig1_ate_forest.png
- fig2_uplift_curves.png
- fig3_overlap_histograms.png
- fig4_policy_gain.png
- fig5_placebo_test.png
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from econml.dml import LinearDML, CausalForestDML
from econml.dr import LinearDRLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SEEDS = [42, 123, 456, 789, 1337, 2024, 31, 99, 7, 404]
CV_FOLDS = 3
N_PLACEBO = 5
MIN_TREATED = 8
MIN_CONTROL = 8


SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR = next(
    (
        p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]
        if (p / "02_ITE" / "01_preprocessing" / "output" / "ite_ready_dataset_v2.csv").exists()
    ),
    None,
)
if BASE_DIR is None:
    raise FileNotFoundError("Could not find ite_ready_dataset_v2.csv from current location.")

INPUT_DIR = BASE_DIR / "02_ITE" / "01_preprocessing" / "output"
OUTPUT_DIR = SCRIPT_DIR / "final_ite_evaluation_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = INPUT_DIR / "ite_ready_dataset_v2.csv"
META_PATH = INPUT_DIR / "preprocessing_metadata_v2.json"
if not META_PATH.exists():
    META_PATH = INPUT_DIR / "preprocessing_metadata.json"

df = pd.read_csv(DATA_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}

drop_cols = [
    "T", "Y", "patient_id", "propensity_score",
    "T_hormone", "T_chemo", "T_targeted", "T_radiation", "T_hormone_excl"
]
X_df = df.drop(columns=drop_cols, errors="ignore")
X_all = X_df.astype(float).values
Y_all = df["Y"].astype(int).values

er = (df["ER_status"].fillna(0).astype(float).values > 0.5)
pr = (df["PR_status"].fillna(0).astype(float).values > 0.5)
her2 = (df["HER2_status"].fillna(0).astype(float).values > 0.5)

T_any = df["T"].astype(int).values if "T" in df.columns else None
T_hormone = df["T_hormone"].astype(int).values if "T_hormone" in df.columns else None
T_chemo = df["T_chemo"].astype(int).values if "T_chemo" in df.columns else None
T_targeted = df["T_targeted"].astype(int).values if "T_targeted" in df.columns else None


def rf_reg(seed: int):
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )


def rf_clf(seed: int):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def propensity_scores(X: np.ndarray, T: np.ndarray, seed: int) -> np.ndarray:
    lr = LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced")
    lr.fit(X, T)
    return lr.predict_proba(X)[:, 1]


def stabilized_ipw(X: np.ndarray, T: np.ndarray, seed: int, clip=(0.05, 0.95)) -> np.ndarray:
    ps = propensity_scores(X, T, seed=seed).clip(*clip)
    p1 = T.mean()
    w = np.where(T == 1, p1 / ps, (1 - p1) / (1 - ps))
    return w / w.mean()


def effective_sample_size(weights: np.ndarray) -> float:
    return (weights.sum() ** 2) / np.sum(weights ** 2)


def standardized_mean_difference(X: np.ndarray, T: np.ndarray, feat_cap: int = 10) -> float:
    smds = []
    for i in range(min(X.shape[1], feat_cap)):
        xt = X[T == 1, i]
        xc = X[T == 0, i]
        pooled = np.sqrt((xt.std() ** 2 + xc.std() ** 2) / 2) + 1e-8
        smds.append(abs((xt.mean() - xc.mean()) / pooled))
    return float(np.mean(smds)) if smds else np.nan


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    return (
        float(np.mean(values)),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def compute_uplift_metrics(ite: np.ndarray, T: np.ndarray, Y: np.ndarray, n_bins: int = 100):
    order = np.argsort(-ite)
    T_ord = T[order]
    Y_ord = Y[order]
    n = len(T_ord)

    fracs, uplifts = [], []
    for k in range(1, n_bins + 1):
        tk = int(n * k / n_bins)
        sT = T_ord[:tk]
        sY = Y_ord[:tk]
        n1 = int(sT.sum())
        n0 = int((sT == 0).sum())
        if n1 == 0 or n0 == 0:
            fracs.append(k / n_bins)
            uplifts.append(np.nan)
            continue
        fracs.append(k / n_bins)
        uplifts.append(float(sY[sT == 1].mean() - sY[sT == 0].mean()))

    fracs = np.asarray(fracs)
    uplifts = np.asarray(uplifts)
    valid = ~np.isnan(uplifts)

    auuc = float(np.trapz(uplifts[valid], fracs[valid])) if valid.any() else np.nan
    raw_gain = float(Y[T == 1].mean() - Y[T == 0].mean()) if (T.sum() > 0 and (T == 0).sum() > 0) else np.nan
    qini = float(auuc - raw_gain * 0.5) if np.isfinite(raw_gain) and np.isfinite(auuc) else np.nan

    return fracs, uplifts, auuc, qini


def compute_policy_metrics(ite: np.ndarray, T: np.ndarray, Y: np.ndarray, X: np.ndarray, seed: int = RANDOM_STATE):
    ps = propensity_scores(X, T, seed=seed).clip(0.05, 0.95)
    ipw1 = np.where(T == 1, Y / ps, 0.0)
    ipw0 = np.where(T == 0, Y / (1 - ps), 0.0)

    policy = (ite < 0).astype(int)
    policy_value = float((policy * ipw1 + (1 - policy) * ipw0).mean())
    treat_all = float(ipw1.mean())
    treat_none = float(ipw0.mean())

    return {
        "policy_value": policy_value,
        "treat_all": treat_all,
        "treat_none": treat_none,
        "policy_gain_vs_none": float(treat_none - policy_value),
        "policy_gain_vs_all": float(treat_all - policy_value),
    }


def summarize_seed_values(values):
    arr = np.asarray(values, dtype=float)
    protective_pct = float(np.mean(arr < 0) * 100)
    return {
        "median": float(np.median(arr)),
        "iqr_lo": float(np.percentile(arr, 25)),
        "iqr_hi": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "protective_pct": protective_pct,
        "stable": bool(protective_pct >= 80 or protective_pct <= 20),
    }


def run_seeded_econml(X: np.ndarray, T: np.ndarray, Y: np.ndarray, sample_weight=None):
    n1 = int(T.sum())
    n0 = int((T == 0).sum())
    if n1 < MIN_TREATED or n0 < MIN_CONTROL:
        return {}

    out = {"LinearDML": [], "CausalForestDML": [], "LinearDRLearner": []}

    for seed in SEEDS:
        cv = min(CV_FOLDS, n1, n0)

        try:
            lin = LinearDML(
                model_y=rf_reg(seed),
                model_t=rf_clf(seed),
                discrete_treatment=True,
                linear_first_stages=False,
                cv=cv,
                random_state=seed,
            )
            kwargs = {"inference": "statsmodels"}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
            lin.fit(Y.astype(float), T, X=X.astype(float), **kwargs)
            out["LinearDML"].append(lin.effect(X.astype(float)).flatten())
        except Exception:
            pass

        try:
            cf = CausalForestDML(
                model_y=rf_reg(seed),
                model_t=rf_clf(seed),
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
            out["CausalForestDML"].append(cf.effect(X.astype(float)).flatten())
        except Exception:
            pass

        try:
            dr = LinearDRLearner(
                model_regression=rf_reg(seed),
                model_propensity=rf_clf(seed),
                cv=cv,
                random_state=seed,
            )
            kwargs = {}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
            dr.fit(Y.astype(float), T, X=X.astype(float), **kwargs)
            out["LinearDRLearner"].append(dr.effect(X.astype(float)).flatten())
        except Exception:
            pass

    return out


arms = []

if T_any is not None:
    arms.append({
        "arm_key": "T_any",
        "arm_label": "Any Treatment",
        "mask": np.ones(len(df), dtype=bool),
        "treatment": T_any,
        "primary": False,
        "exploratory": False,
    })

if T_hormone is not None:
    arms.append({
        "arm_key": "T_hormone",
        "arm_label": "Hormone Therapy",
        "mask": ((er | pr) & (~her2)),
        "treatment": T_hormone,
        "primary": True,
        "exploratory": False,
    })

if T_chemo is not None:
    arms.append({
        "arm_key": "T_chemo",
        "arm_label": "Chemotherapy",
        "mask": ((~er) & (~pr) & (~her2)),
        "treatment": T_chemo,
        "primary": True,
        "exploratory": False,
    })

if T_targeted is not None and T_chemo is not None:
    arms.append({
        "arm_key": "T_targeted",
        "arm_label": "Targeted Therapy",
        "mask": (her2 & (T_chemo == 1)),
        "treatment": T_targeted,
        "primary": False,
        "exploratory": True,
    })


results_long = []
seed_rows = []
overlap_rows = []
uplift_store = {}
policy_rows = []
placebo_rows = []

for arm in arms:
    mask = arm["mask"].astype(bool)
    X = X_all[mask]
    Y = Y_all[mask]
    T = arm["treatment"][mask].astype(int)

    n = len(Y)
    n1 = int(T.sum())
    n0 = int((T == 0).sum())
    if n1 < MIN_TREATED or n0 < MIN_CONTROL:
        continue

    variants = [("baseline", None, np.ones(n, dtype=bool))]

    ps_vals = propensity_scores(X, T, RANDOM_STATE)
    trim_keep = (ps_vals >= 0.10) & (ps_vals <= 0.90)
    if trim_keep.sum() > 20 and int(T[trim_keep].sum()) >= MIN_TREATED and int((T[trim_keep] == 0).sum()) >= MIN_CONTROL:
        variants.append(("ps_trim_10_90", None, trim_keep))

    variants.append(("stabilized_ipw", stabilized_ipw(X, T, RANDOM_STATE), np.ones(n, dtype=bool)))

    for variant_name, variant_weights, keep in variants:
        Xv = X[keep]
        Yv = Y[keep]
        Tv = T[keep]
        sw = variant_weights[keep] if (variant_weights is not None and len(variant_weights) == len(keep)) else variant_weights

        overlap_rows.append({
            "arm_key": arm["arm_key"],
            "arm_label": arm["arm_label"],
            "variant": variant_name,
            "n": int(len(Yv)),
            "n_treated": int(Tv.sum()),
            "n_control": int((Tv == 0).sum()),
            "ps_min": float(propensity_scores(Xv, Tv, RANDOM_STATE).min()),
            "ps_max": float(propensity_scores(Xv, Tv, RANDOM_STATE).max()),
            "max_ipw": float(stabilized_ipw(Xv, Tv, RANDOM_STATE).max()),
            "ess": float(effective_sample_size(stabilized_ipw(Xv, Tv, RANDOM_STATE))),
            "mean_smd": standardized_mean_difference(Xv, Tv),
        })

        model_outputs = run_seeded_econml(Xv, Tv, Yv, sample_weight=sw)
        for model_name, ite_list in model_outputs.items():
            if len(ite_list) == 0:
                continue

            ate_values = [float(np.mean(ite)) for ite in ite_list]
            att_values = [float(np.mean(ite[Tv == 1])) for ite in ite_list]
            ate_summary = summarize_seed_values(ate_values)
            att_summary = summarize_seed_values(att_values)

            for seed, ite in zip(SEEDS[:len(ite_list)], ite_list):
                seed_rows.append({
                    "arm_key": arm["arm_key"],
                    "arm_label": arm["arm_label"],
                    "variant": variant_name,
                    "model": model_name,
                    "seed": seed,
                    "ATE": float(np.mean(ite)),
                    "ATT": float(np.mean(ite[Tv == 1])),
                    "pct_benefit": float(np.mean(ite < 0) * 100),
                })

            median_idx = int(np.argmin(np.abs(np.asarray(ate_values) - ate_summary["median"])))
            ite_ref = ite_list[median_idx]

            fracs, uplifts, auuc, qini = compute_uplift_metrics(ite_ref, Tv, Yv)
            policy = compute_policy_metrics(ite_ref, Tv, Yv, Xv, seed=RANDOM_STATE)

            uplift_store[(arm["arm_key"], variant_name, model_name)] = {
                "fracs": fracs,
                "uplifts": uplifts,
            }

            policy_rows.append({
                "arm_key": arm["arm_key"],
                "arm_label": arm["arm_label"],
                "variant": variant_name,
                "model": model_name,
                **policy,
            })

            ate_mean, ate_lo, ate_hi = bootstrap_mean_ci(ite_ref, seed=RANDOM_STATE)

            results_long.append({
                "arm_key": arm["arm_key"],
                "arm_label": arm["arm_label"],
                "primary": arm["primary"],
                "exploratory": arm["exploratory"],
                "variant": variant_name,
                "model": model_name,
                "n": int(len(Yv)),
                "n_treated": int(Tv.sum()),
                "n_control": int((Tv == 0).sum()),
                "ATE_boot_mean": ate_mean,
                "ATE_boot_lo": ate_lo,
                "ATE_boot_hi": ate_hi,
                "ATE_median_seed": ate_summary["median"],
                "ATE_iqr_lo": ate_summary["iqr_lo"],
                "ATE_iqr_hi": ate_summary["iqr_hi"],
                "ATE_min_seed": ate_summary["min"],
                "ATE_max_seed": ate_summary["max"],
                "ATE_protective_pct": ate_summary["protective_pct"],
                "ATE_stable": ate_summary["stable"],
                "ATT_median_seed": att_summary["median"],
                "ATT_iqr_lo": att_summary["iqr_lo"],
                "ATT_iqr_hi": att_summary["iqr_hi"],
                "AUUC": auuc,
                "Qini": qini,
                "policy_value": policy["policy_value"],
                "treat_all": policy["treat_all"],
                "treat_none": policy["treat_none"],
                "policy_gain_vs_none": policy["policy_gain_vs_none"],
                "policy_gain_vs_all": policy["policy_gain_vs_all"],
                "pct_benefit_ref": float(np.mean(ite_ref < 0) * 100),
                "ITE_std_ref": float(np.std(ite_ref)),
                "ITE_iqr_ref": float(np.percentile(ite_ref, 75) - np.percentile(ite_ref, 25)),
            })

    for shuffle_idx in range(1, N_PLACEBO + 1):
        rng = np.random.default_rng(RANDOM_STATE + shuffle_idx)
        Y_dummy = rng.permutation(Y)
        placebo_outputs = run_seeded_econml(X, T, Y_dummy)
        for model_name, ite_list in placebo_outputs.items():
            if len(ite_list) == 0:
                continue
            ate_vals = [float(np.mean(ite)) for ite in ite_list]
            summary = summarize_seed_values(ate_vals)
            placebo_rows.append({
                "arm_key": arm["arm_key"],
                "arm_label": arm["arm_label"],
                "shuffle": shuffle_idx,
                "model": model_name,
                "median_ATE_placebo": summary["median"],
                "ATE_min_placebo": summary["min"],
                "ATE_max_placebo": summary["max"],
                "protective_pct_placebo": summary["protective_pct"],
            })


results_df = pd.DataFrame(results_long)
seed_df = pd.DataFrame(seed_rows)
overlap_df = pd.DataFrame(overlap_rows)
placebo_df = pd.DataFrame(placebo_rows)
policy_df = pd.DataFrame(policy_rows)

results_df.to_csv(OUTPUT_DIR / "final_ite_results_long.csv", index=False)
seed_df.to_csv(OUTPUT_DIR / "final_ite_seedwise_ate.csv", index=False)
overlap_df.to_csv(OUTPUT_DIR / "final_ite_overlap.csv", index=False)
placebo_df.to_csv(OUTPUT_DIR / "final_ite_placebo.csv", index=False)
policy_df.to_csv(OUTPUT_DIR / "final_ite_policy.csv", index=False)

wide_cols = [
    "arm_label", "variant", "model", "n_treated", "n_control",
    "ATE_median_seed", "ATE_iqr_lo", "ATE_iqr_hi", "AUUC", "Qini",
    "policy_gain_vs_none", "ATE_protective_pct", "ATE_stable", "exploratory"
]
results_df[wide_cols].to_csv(OUTPUT_DIR / "final_ite_results_wide.csv", index=False)

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

model_order = ["LinearDML", "CausalForestDML", "LinearDRLearner"]
variant_order = ["baseline", "stabilized_ipw", "ps_trim_10_90"]

fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 6), squeeze=False)
axes = axes[0]
for ax, arm in zip(axes, arms):
    sub = results_df[
        (results_df["arm_key"] == arm["arm_key"]) &
        (results_df["variant"].isin(variant_order))
    ].copy()
    sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
    sub["variant"] = pd.Categorical(sub["variant"], categories=variant_order, ordered=True)
    sub = sub.sort_values(["variant", "model"]).reset_index(drop=True)
    labels = [f"{v} | {m}" for v, m in zip(sub["variant"], sub["model"])]
    y = np.arange(len(sub))
    vals = sub["ATE_median_seed"].values
    lo = sub["ATE_iqr_lo"].values
    hi = sub["ATE_iqr_hi"].values
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", capsize=4)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(arm["arm_label"] + (" (exploratory)" if arm["exploratory"] else ""))
    ax.set_xlabel("Median ATE with IQR across seeds")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_ate_forest.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 5), squeeze=False)
axes = axes[0]
for ax, arm in zip(axes, arms):
    for model_name in model_order:
        key = (arm["arm_key"], "baseline", model_name)
        if key not in uplift_store:
            continue
        d = uplift_store[key]
        valid = ~np.isnan(d["uplifts"])
        ax.plot(d["fracs"][valid], d["uplifts"][valid], linewidth=2, label=model_name)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(arm["arm_label"] + " uplift")
    ax.set_xlabel("Fraction targeted by ITE rank")
    ax.set_ylabel("Observed uplift")
    ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_uplift_curves.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), squeeze=False)
axes = axes[0]
for ax, arm in zip(axes, arms):
    mask = arm["mask"].astype(bool)
    X = X_all[mask]
    T = arm["treatment"][mask].astype(int)
    ps = propensity_scores(X, T, RANDOM_STATE)
    ax.hist(ps[T == 1], bins=20, alpha=0.65, density=True, label="Treated")
    ax.hist(ps[T == 0], bins=20, alpha=0.65, density=True, label="Control")
    ax.axvline(0.10, linestyle="--", linewidth=1)
    ax.axvline(0.90, linestyle="--", linewidth=1)
    ax.set_title(arm["arm_label"])
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_overlap_histograms.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 5), squeeze=False)
axes = axes[0]
for ax, arm in zip(axes, arms):
    sub = results_df[
        (results_df["arm_key"] == arm["arm_key"]) &
        (results_df["variant"] == "baseline")
    ].copy()
    sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
    sub = sub.sort_values("model")
    ax.bar(sub["model"], sub["policy_gain_vs_none"])
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(arm["arm_label"])
    ax.set_ylabel("Policy gain vs treat-none")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_policy_gain.png", bbox_inches="tight")
plt.close(fig)

if len(placebo_df) > 0:
    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 5), squeeze=False)
    axes = axes[0]
    for ax, arm in zip(axes, arms):
        sub = placebo_df[placebo_df["arm_key"] == arm["arm_key"]].copy()
        if sub.empty:
            ax.axis("off")
            continue
        data = [sub[sub["model"] == m]["median_ATE_placebo"].values for m in model_order if m in sub["model"].unique()]
        labels = [m for m in model_order if m in sub["model"].unique()]
        ax.boxplot(data, labels=labels)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(arm["arm_label"])
        ax.set_ylabel("Placebo median ATE")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_placebo_test.png", bbox_inches="tight")
    plt.close(fig)

summary = {
    "dataset": str(DATA_PATH),
    "n_patients": int(len(df)),
    "n_features": int(X_all.shape[1]),
    "n_events": int(Y_all.sum()),
    "event_rate": float(Y_all.mean()),
    "arms_reported": sorted(results_df["arm_label"].unique().tolist()) if len(results_df) else [],
    "notes": [
        "Use AUUC, Qini, and Policy Gain for model-selection reporting.",
        "Use repeated-seed ATE summaries, overlap diagnostics, and placebo tests for final robustness reporting.",
        "Keep targeted therapy exploratory due to small treated sample size."
    ],
}
(OUTPUT_DIR / "final_ite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"Saved outputs to: {OUTPUT_DIR}")