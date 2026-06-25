#!/usr/bin/env python3
"""
Reviewer 3 pipeline-level MB ablation for TCGA-BRCA multimodal prognosis.

Purpose
-------
This script addresses the reviewer request more strongly than the fixed-signature
ablation by RE-RUNNING feature selection on the candidate dataset, not only on the
already selected final 20 features.

It compares:
  1) Clinical-only pipeline
  2) Omics-only pipeline
  3) Clinical + omics pipeline

For each feature pool, the script:
  - loads the Stage-2 candidate matrix, defaulting to:
      ../MERGE_continuous_outer/08_composite.csv
  - defines the candidate feature pool by modality prefixes;
  - runs a repository-compatible Markov Blanket-style local discovery approximation
    based on mutual information with a permutation threshold, using the same
    algorithm names used in run_mb_multimodal.py: IAMB, GSMB, MMMB;
  - uses fallback augmentation if the MB neighborhood is too small;
  - optionally refines/caps the selected set to TOP_N features using univariate
    penalized Cox ranking, mirroring the final top-20 survival-modeling strategy;
  - evaluates Cox survival models with 5-fold cross-validation;
  - tunes/evaluates a Cox penalizer grid fairly for all three pools.

Default assumptions match the repository workflow:
  - script location: 01_Causal_feature_extraction/MB/
  - input dataset: 01_Causal_feature_extraction/MERGE_continuous_outer/08_composite.csv
  - outcome columns: OS, OS.time
  - algorithms: IAMB, GSMB, MMMB
  - alphas: 0.05, 0.10, 0.20
  - top-N: 20
  - CV: 5-fold, random_state=42

Outputs
-------
Creates MB/results_reviewer3_pipeline_mb_ablation/ with:
  - reviewer3_pipeline_all_configs.csv
  - reviewer3_pipeline_fold_metrics.csv
  - reviewer3_pipeline_best_by_pool.csv
  - reviewer3_pipeline_fixed_multimodal_config.csv
  - reviewer3_pipeline_selected_features.csv
  - reviewer3_pipeline_table_for_manuscript.csv
  - reviewer3_pipeline_table_for_manuscript.tex
  - reviewer3_pipeline_report.txt
  - reviewer3_pipeline_barplot.png
  - reviewer3_pipeline_run_metadata.json

Run
---
cd /path/to/Thesis_v3/01_Causal_feature_extraction/MB
python reviewer3_pipeline_mb_ablation.py

Optional examples:
python reviewer3_pipeline_mb_ablation.py --top-n 20 --penalizer-grid 0.1 0.5 1 2 5 10
python reviewer3_pipeline_mb_ablation.py --scale-for-cox
python reviewer3_pipeline_mb_ablation.py --data-file ../MERGE_continuous_outer/08_composite.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CLINICAL_PREFIX = "CLIN_"
OMICS_PREFIXES = ("RNA_", "CNV_", "MUT_", "PROT_", "METH_", "MIRNA_")
ALL_MODALITY_PREFIXES = (CLINICAL_PREFIX,) + OMICS_PREFIXES
DEFAULT_ALGORITHMS = ("IAMB", "GSMB", "MMMB")
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)

# Simple in-run caches so MI/permutation and dCor are computed once per pool/alpha
# instead of once per algorithm. This keeps the script practical on Windows laptops.
_MB_SCORE_CACHE: Dict[Tuple[Tuple[str, ...], int, float, int, int], Tuple[Dict[str, float], Dict[str, float]]] = {}
_COMP_RANK_CACHE: Dict[Tuple[Tuple[str, ...], int, int, bool], Dict[str, float]] = {}


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def infer_project_dir() -> Path:
    """Infer the 01_Causal_feature_extraction directory from script/cwd."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd().resolve()

    candidates = _unique_paths([here, Path.cwd().resolve()] + list(here.parents) + list(Path.cwd().resolve().parents))
    for p in candidates:
        if p.name == "01_Causal_feature_extraction" and (p / "MB").exists():
            return p
        if (p / "01_Causal_feature_extraction" / "MB").exists():
            return p / "01_Causal_feature_extraction"
        if (p / "MB").exists() and ((p / "MERGE_continuous_outer").exists() or (p / "MERGE").exists()):
            return p

    if here.name == "MB":
        return here.parent
    return here


def find_dataset_file(project_dir: Path, dataset_name: str) -> Optional[Path]:
    filename = f"{dataset_name}.csv"
    candidates = [
        project_dir / "MERGE_continuous_outer" / filename,
        project_dir / "MERGE" / filename,
        project_dir / filename,
        project_dir / "MB" / "MERGE_continuous_outer" / filename,
        Path.cwd() / filename,
        Path.cwd() / "MERGE_continuous_outer" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    search_roots = _unique_paths([project_dir, project_dir.parent, Path.cwd().resolve()])
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(filename))
        matches = [m for m in matches if "selected_features" not in str(m)]
        if matches:
            return matches[0].resolve()
    return None


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def read_merged_dataset(path: Path, event_col: str, duration_col: str) -> pd.DataFrame:
    """Read CSV robustly. Existing code generally uses index_col=0, but fallback if needed."""
    df = pd.read_csv(path, index_col=0)
    if event_col not in df.columns or duration_col not in df.columns:
        df = pd.read_csv(path)

    missing = [c for c in (event_col, duration_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Outcome columns not found in {path}: {missing}. "
            f"Available columns start with: {list(df.columns[:20])}"
        )

    df = df.copy()
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df = df.dropna(subset=[event_col, duration_col])
    df = df[df[duration_col] > 0].copy()
    return df


def feature_columns(df: pd.DataFrame, event_col: str, duration_col: str) -> List[str]:
    excluded = {event_col, duration_col}
    return [c for c in df.columns if c not in excluded]


def modality_of(feature: str) -> str:
    for prefix in ALL_MODALITY_PREFIXES:
        if feature.startswith(prefix):
            return prefix.rstrip("_")
    return "OTHER"


def is_clinical_feature(feature: str) -> bool:
    return feature.startswith(CLINICAL_PREFIX)


def is_omics_feature(feature: str) -> bool:
    return feature.startswith(OMICS_PREFIXES)


def coerce_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def drop_unusable_features(X: pd.DataFrame, min_nonmissing: int = 10) -> pd.DataFrame:
    """Remove all-missing, mostly-missing, or constant columns."""
    keep = []
    for c in X.columns:
        s = X[c]
        if int(s.notna().sum()) < min_nonmissing:
            continue
        vals = s.dropna()
        if vals.empty:
            continue
        if float(vals.std(ddof=0)) == 0.0:
            continue
        keep.append(c)
    return X[keep].copy()


def median_impute_fit_transform(X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.Series]:
    med = X_train.median(axis=0, skipna=True)
    med = med.fillna(0.0)
    Xtr = X_train.fillna(med).astype(float)
    if X_test is None:
        return Xtr, None, med
    Xte = X_test.fillna(med).astype(float)
    return Xtr, Xte, med


def scale_fit_transform(X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    scaler = StandardScaler()
    Xtr_arr = scaler.fit_transform(X_train.values.astype(float))
    Xtr = pd.DataFrame(Xtr_arr, index=X_train.index, columns=X_train.columns)
    if X_test is None:
        return Xtr, None
    Xte_arr = scaler.transform(X_test.values.astype(float))
    Xte = pd.DataFrame(Xte_arr, index=X_test.index, columns=X_test.columns)
    return Xtr, Xte


# -----------------------------------------------------------------------------
# Ranking and MB-style discovery helpers
# -----------------------------------------------------------------------------

def rank_desc(values: Dict[str, float]) -> Dict[str, float]:
    """Rank higher values better. Missing/nonfinite values get worst rank."""
    clean = {k: (float(v) if np.isfinite(v) else -np.inf) for k, v in values.items()}
    ordered = sorted(clean, key=lambda k: clean[k], reverse=True)
    return {k: float(i + 1) for i, k in enumerate(ordered)}


def rank_asc(values: Dict[str, float]) -> Dict[str, float]:
    """Rank lower values better. Missing/nonfinite values get worst rank."""
    clean = {k: (float(v) if np.isfinite(v) else np.inf) for k, v in values.items()}
    ordered = sorted(clean, key=lambda k: clean[k])
    return {k: float(i + 1) for i, k in enumerate(ordered)}


def spearman_scores(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y_rank = y.rank(method="average")
    for c in X.columns:
        try:
            r = X[c].rank(method="average").corr(y_rank)
            out[c] = abs(float(r)) if np.isfinite(r) else 0.0
        except Exception:
            out[c] = 0.0
    return out


def distance_correlation_1d(x: np.ndarray, y_centered_dist: np.ndarray) -> float:
    """Distance correlation for one feature against a pre-centered y distance matrix."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x[:, None] - x[None, :])
    a = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b = y_centered_dist
    dcov2 = np.mean(a * b)
    dvarx = np.mean(a * a)
    dvary = np.mean(b * b)
    if dvarx <= 0 or dvary <= 0:
        return 0.0
    val = math.sqrt(max(dcov2, 0.0) / math.sqrt(dvarx * dvary))
    return float(val) if np.isfinite(val) else 0.0


def distance_correlation_scores(
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int = 42,
    max_samples: int = 500,
) -> Dict[str, float]:
    """Approximate distance correlation scores on a fixed subsample if needed."""
    n = len(X)
    if n == 0:
        return {c: 0.0 for c in X.columns}
    if n > max_samples:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        X_use = X.iloc[idx]
        y_use = y.iloc[idx]
    else:
        X_use = X
        y_use = y

    y_arr = y_use.values.astype(float)
    b = np.abs(y_arr[:, None] - y_arr[None, :])
    b = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    scores: Dict[str, float] = {}
    for c in X_use.columns:
        try:
            scores[c] = distance_correlation_1d(X_use[c].values.astype(float), b)
        except Exception:
            scores[c] = 0.0
    return scores


def compute_mi_scores_and_thresholds(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    n_perm: int,
    random_seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """MI score and per-feature permutation threshold."""
    if X.empty:
        return {}, {}

    X_imp, _, _ = median_impute_fit_transform(X)
    X_scaled, _ = scale_fit_transform(X_imp)
    X_arr = X_scaled.values.astype(float)
    y_arr = y.values.astype(float)

    mi = mutual_info_regression(X_arr, y_arr, random_state=random_seed)

    rng = np.random.default_rng(random_seed)
    null = []
    for p in range(n_perm):
        y_perm = rng.permutation(y_arr)
        null.append(mutual_info_regression(X_arr, y_perm, random_state=random_seed + p + 1))
    null_arr = np.vstack(null) if null else np.zeros((1, X_arr.shape[1]))
    threshold = np.percentile(null_arr, (1.0 - alpha) * 100.0, axis=0)

    mi_scores = {c: float(mi[i]) if np.isfinite(mi[i]) else 0.0 for i, c in enumerate(X.columns)}
    thresholds = {c: float(threshold[i]) if np.isfinite(threshold[i]) else 0.0 for i, c in enumerate(X.columns)}
    return mi_scores, thresholds


def compute_composite_fallback_ranks(
    X: pd.DataFrame,
    y_time: pd.Series,
    mi_scores: Dict[str, float],
    dcor_max_samples: int,
    random_seed: int,
    skip_dcor: bool = False,
) -> Dict[str, float]:
    """
    Composite fallback rank inspired by the manuscript formula:
      average rank of |Spearman|, mutual information, and distance correlation.
    Lower composite rank is better.
    """
    sp = spearman_scores(X, y_time)
    if skip_dcor:
        dcor = {c: 0.0 for c in X.columns}
    else:
        dcor = distance_correlation_scores(X, y_time, random_seed=random_seed, max_samples=dcor_max_samples)

    r_sp = rank_desc(sp)
    r_mi = rank_desc(mi_scores)
    r_dc = rank_desc(dcor)

    comp = {}
    for c in X.columns:
        comp[c] = (r_sp.get(c, len(X.columns)) + r_mi.get(c, len(X.columns)) + r_dc.get(c, len(X.columns))) / 3.0
    return comp


def compute_global_cox_pvals(
    df: pd.DataFrame,
    features: Sequence[str],
    event_col: str,
    duration_col: str,
    penalizer: float,
    scale_for_cox: bool,
) -> Dict[str, float]:
    """Univariate Cox p-values for survival-aware final ranking. Lower is better."""
    out: Dict[str, float] = {}
    base = df[[event_col, duration_col]].copy()
    for i, f in enumerate(features, start=1):
        try:
            X = coerce_numeric_frame(df, [f])
            X = drop_unusable_features(X, min_nonmissing=10)
            if f not in X.columns:
                out[f] = 1.0
                continue
            X_imp, _, _ = median_impute_fit_transform(X)
            if scale_for_cox:
                X_imp, _ = scale_fit_transform(X_imp)
            sub = pd.concat([X_imp, base], axis=1)
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(sub[[f, event_col, duration_col]], duration_col=duration_col, event_col=event_col)
            p = float(cph.summary["p"].iloc[0])
            out[f] = p if np.isfinite(p) else 1.0
        except Exception:
            out[f] = 1.0
    return out


@dataclass
class MBSelectionResult:
    pool_name: str
    algorithm: str
    alpha: float
    pool_n_features: int
    mb_initial_n: int
    fallback_added_n: int
    mb_candidate_n: int
    selected_features: List[str]
    mb_features: List[str]
    fallback_features: List[str]
    mi_scores: Dict[str, float]
    composite_ranks: Dict[str, float]


def run_repo_compatible_mb(
    X: pd.DataFrame,
    y_time: pd.Series,
    algorithm: str,
    alpha: float,
    n_perm: int,
    random_seed: int,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """
    Repository-compatible Python MB approximation.

    This follows the structure of run_mb_multimodal.py:
      - standardize features;
      - compute mutual_info_regression with OS.time;
      - compute permutation threshold;
      - select MI > threshold;
      - apply algorithm-specific refinement for GSMB/MMMB.
    """
    cache_key = (tuple(X.columns), len(X), float(alpha), int(n_perm), int(random_seed))
    if cache_key in _MB_SCORE_CACHE:
        mi_scores, thresholds = _MB_SCORE_CACHE[cache_key]
    else:
        mi_scores, thresholds = compute_mi_scores_and_thresholds(
            X, y_time, alpha=alpha, n_perm=n_perm, random_seed=random_seed
        )
        _MB_SCORE_CACHE[cache_key] = (mi_scores, thresholds)

    selected = [c for c in X.columns if mi_scores.get(c, 0.0) > thresholds.get(c, np.inf)]
    selected = sorted(selected, key=lambda c: mi_scores.get(c, 0.0), reverse=True)

    if algorithm.upper() == "IAMB":
        pass
    elif algorithm.upper() == "GSMB":
        selected = selected[:60]
    elif algorithm.upper() == "MMMB":
        if selected:
            vals = np.array([mi_scores.get(c, 0.0) for c in selected], dtype=float)
            cutoff = float(np.percentile(vals, 25))
            selected = [c for c in selected if mi_scores.get(c, 0.0) >= cutoff]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use IAMB, GSMB, or MMMB.")

    return selected, mi_scores, thresholds


def select_pipeline_features(
    df: pd.DataFrame,
    pool_name: str,
    pool_features: Sequence[str],
    algorithm: str,
    alpha: float,
    event_col: str,
    duration_col: str,
    top_n: int,
    mb_candidate_cap: int,
    min_mb_features: int,
    n_perm: int,
    random_seed: int,
    cox_pvals: Dict[str, float],
    dcor_max_samples: int,
    skip_dcor: bool,
    final_ranker: str,
) -> MBSelectionResult:
    X = coerce_numeric_frame(df, pool_features)
    X = drop_unusable_features(X, min_nonmissing=10)
    if X.empty:
        raise ValueError(f"No usable features in pool {pool_name}.")

    y_time = df[duration_col].astype(float)

    mb_raw, mi_scores, _thresholds = run_repo_compatible_mb(
        X=X,
        y_time=y_time,
        algorithm=algorithm,
        alpha=alpha,
        n_perm=n_perm,
        random_seed=random_seed,
    )
    mb_initial_n = len(mb_raw)

    comp_key = (tuple(X.columns), len(X), int(dcor_max_samples), bool(skip_dcor))
    if comp_key in _COMP_RANK_CACHE:
        comp_ranks = _COMP_RANK_CACHE[comp_key]
    else:
        comp_ranks = compute_composite_fallback_ranks(
            X=X,
            y_time=y_time,
            mi_scores=mi_scores,
            dcor_max_samples=dcor_max_samples,
            random_seed=random_seed,
            skip_dcor=skip_dcor,
        )
        _COMP_RANK_CACHE[comp_key] = comp_ranks

    # Fallback augmentation if MB is too sparse.
    fallback_features: List[str] = []
    target_min = min(min_mb_features, X.shape[1])
    selected_candidate = list(dict.fromkeys(mb_raw))
    if len(selected_candidate) < target_min:
        ranked_fallback = sorted(X.columns, key=lambda c: comp_ranks.get(c, np.inf))
        for f in ranked_fallback:
            if f not in selected_candidate:
                selected_candidate.append(f)
                fallback_features.append(f)
            if len(selected_candidate) >= target_min:
                break

    # Candidate cap before final survival-aware refinement.
    # Keep highest MI / best composite candidates.
    if len(selected_candidate) > mb_candidate_cap:
        if final_ranker == "composite":
            selected_candidate = sorted(selected_candidate, key=lambda c: comp_ranks.get(c, np.inf))[:mb_candidate_cap]
        else:
            # Keep candidates with strongest Cox p-values before final top-N.
            selected_candidate = sorted(selected_candidate, key=lambda c: cox_pvals.get(c, 1.0))[:mb_candidate_cap]

    # Final selected features: cap to top-N for fair model-size comparison.
    final_n = min(top_n, len(selected_candidate))
    if final_ranker == "cox":
        final_selected = sorted(selected_candidate, key=lambda c: cox_pvals.get(c, 1.0))[:final_n]
    elif final_ranker == "composite":
        final_selected = sorted(selected_candidate, key=lambda c: comp_ranks.get(c, np.inf))[:final_n]
    elif final_ranker == "mi":
        final_selected = sorted(selected_candidate, key=lambda c: mi_scores.get(c, 0.0), reverse=True)[:final_n]
    else:
        raise ValueError("final_ranker must be one of: cox, composite, mi")

    return MBSelectionResult(
        pool_name=pool_name,
        algorithm=algorithm,
        alpha=alpha,
        pool_n_features=int(X.shape[1]),
        mb_initial_n=int(mb_initial_n),
        fallback_added_n=int(len(fallback_features)),
        mb_candidate_n=int(len(selected_candidate)),
        selected_features=list(final_selected),
        mb_features=list(mb_raw),
        fallback_features=list(fallback_features),
        mi_scores=mi_scores,
        composite_ranks=comp_ranks,
    )


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------

def cindex_manual(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    n_conc = n_disc = n_tied = 0
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 0 and events[j] == 0:
                continue
            if times[i] == times[j]:
                continue
            early, late = (i, j) if times[i] < times[j] else (j, i)
            if not events[early]:
                continue
            if scores[early] > scores[late]:
                n_conc += 1
            elif scores[early] < scores[late]:
                n_disc += 1
            else:
                n_tied += 1
    total = n_conc + n_disc + n_tied
    return (n_conc + 0.5 * n_tied) / total if total > 0 else 0.5


def prepare_fold_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    scale_for_cox: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = coerce_numeric_frame(train_df, features)
    X_test = coerce_numeric_frame(test_df, features)
    Xtr, Xte, _ = median_impute_fit_transform(X_train, X_test)
    assert Xte is not None
    if scale_for_cox:
        Xtr, Xte = scale_fit_transform(Xtr, Xte)
        assert Xte is not None
    return Xtr, Xte


def evaluate_cvx_cox(
    df: pd.DataFrame,
    features: Sequence[str],
    event_col: str,
    duration_col: str,
    penalizer: float,
    n_folds: int,
    random_seed: int,
    survival_years: float,
    scale_for_cox: bool,
    auc_mode: str,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """5-fold CV Cox C-index and 5-year AUC.

    auc_mode:
      - logistic: repository-compatible 5-year classification AUC on selected features.
      - risk: AUC using Cox risk score on patients evaluable at 5 years.
      - both: main auc_5yr is risk; logistic also reported.
    """
    if not features:
        return {
            "c_index_mean": 0.5,
            "c_index_sd": 0.0,
            "auc_5yr_mean": 0.5,
            "auc_5yr_sd": 0.0,
            "auc_5yr_risk_mean": 0.5,
            "auc_5yr_risk_sd": 0.0,
            "auc_5yr_logistic_mean": 0.5,
            "auc_5yr_logistic_sd": 0.0,
        }, []

    sub = df[list(features) + [event_col, duration_col]].copy()
    sub[event_col] = pd.to_numeric(sub[event_col], errors="coerce")
    sub[duration_col] = pd.to_numeric(sub[duration_col], errors="coerce")
    sub = sub.dropna(subset=[event_col, duration_col])
    sub = sub[sub[duration_col] > 0].copy()

    if len(sub) < max(40, n_folds * 10):
        raise ValueError(f"Too few usable rows for CV: n={len(sub)}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    threshold = float(survival_years) * 365.0

    fold_rows: List[Dict[str, float]] = []
    c_scores: List[float] = []
    auc_risk_scores: List[float] = []
    auc_logistic_scores: List[float] = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(sub), start=1):
        train = sub.iloc[tr_idx].copy()
        test = sub.iloc[te_idx].copy()
        Xtr, Xte = prepare_fold_matrices(train, test, features, scale_for_cox=scale_for_cox)

        train_fit = Xtr.reset_index(drop=True).copy()
        test_fit = Xte.reset_index(drop=True).copy()
        train_fit[event_col] = train[event_col].values
        train_fit[duration_col] = train[duration_col].values
        test_fit[event_col] = test[event_col].values
        test_fit[duration_col] = test[duration_col].values

        # Cox fit and C-index/risk-score AUC.
        fold_c = 0.5
        risk = np.zeros(len(test_fit), dtype=float)
        cox_ok = False
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(train_fit[list(features) + [event_col, duration_col]], duration_col=duration_col, event_col=event_col)
            risk = cph.predict_partial_hazard(test_fit[list(features)]).values.astype(float)
            fold_c = cindex_manual(
                test_fit[duration_col].values.astype(float),
                test_fit[event_col].values.astype(int),
                risk,
            )
            cox_ok = True
        except Exception:
            # Last-resort fallback: average standardized feature vector.
            try:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr.values.astype(float))
                Xte_s = scaler.transform(Xte.values.astype(float))
                risk = Xte_s.mean(axis=1)
                fold_c = cindex_manual(
                    test_fit[duration_col].values.astype(float),
                    test_fit[event_col].values.astype(int),
                    risk,
                )
            except Exception:
                fold_c = 0.5
                risk = np.zeros(len(test_fit), dtype=float)

        c_scores.append(float(fold_c))

        # AUC based on Cox risk at 5-year status, excluding censored before threshold.
        fold_auc_risk = np.nan
        try:
            evaluable = (test_fit[duration_col] >= threshold) | (test_fit[event_col] == 1)
            te = test_fit.loc[evaluable].copy()
            risk_eval = np.asarray(risk)[np.where(evaluable.values)[0]]
            yte = ((te[event_col] == 1) & (te[duration_col] <= threshold)).astype(int)
            if yte.sum() >= 1 and (1 - yte).sum() >= 1:
                fold_auc_risk = float(roc_auc_score(yte, risk_eval))
                auc_risk_scores.append(fold_auc_risk)
        except Exception:
            pass

        # Repository-compatible 5-year AUC: train a logistic model within the same fold.
        fold_auc_log = np.nan
        if auc_mode in ("logistic", "both"):
            try:
                s5_train = train[(train[duration_col] >= threshold) | (train[event_col] == 1)].copy()
                s5_test = test[(test[duration_col] >= threshold) | (test[event_col] == 1)].copy()
                ytr = ((s5_train[event_col] == 1) & (s5_train[duration_col] <= threshold)).astype(int)
                yte2 = ((s5_test[event_col] == 1) & (s5_test[duration_col] <= threshold)).astype(int)
                if ytr.sum() >= 3 and (1 - ytr).sum() >= 3 and yte2.sum() >= 1 and (1 - yte2).sum() >= 1:
                    Xtr_l, Xte_l = prepare_fold_matrices(s5_train, s5_test, features, scale_for_cox=False)
                    scaler_l = StandardScaler()
                    Xtr_arr = scaler_l.fit_transform(Xtr_l.values.astype(float))
                    Xte_arr = scaler_l.transform(Xte_l.values.astype(float))
                    lr = LogisticRegression(C=0.1, max_iter=500, random_state=random_seed)
                    lr.fit(Xtr_arr, ytr.values.astype(int))
                    prob = lr.predict_proba(Xte_arr)[:, 1]
                    fold_auc_log = float(roc_auc_score(yte2, prob))
                    auc_logistic_scores.append(fold_auc_log)
            except Exception:
                pass

        fold_rows.append(
            {
                "fold": fold,
                "penalizer": float(penalizer),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_features": int(len(features)),
                "cox_fit_ok": bool(cox_ok),
                "c_index": float(fold_c),
                "auc_5yr_risk": float(fold_auc_risk) if np.isfinite(fold_auc_risk) else np.nan,
                "auc_5yr_logistic": float(fold_auc_log) if np.isfinite(fold_auc_log) else np.nan,
            }
        )

    def mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
        vals2 = [float(v) for v in vals if np.isfinite(v)]
        if not vals2:
            return 0.5, 0.0
        return float(np.mean(vals2)), float(np.std(vals2, ddof=1)) if len(vals2) > 1 else 0.0

    c_mean, c_sd = mean_sd(c_scores)
    risk_mean, risk_sd = mean_sd(auc_risk_scores)
    log_mean, log_sd = mean_sd(auc_logistic_scores)

    if auc_mode == "risk":
        auc_main_mean, auc_main_sd = risk_mean, risk_sd
    elif auc_mode == "logistic":
        auc_main_mean, auc_main_sd = log_mean, log_sd
    else:  # both
        auc_main_mean, auc_main_sd = risk_mean, risk_sd

    summary = {
        "c_index_mean": c_mean,
        "c_index_sd": c_sd,
        "auc_5yr_mean": auc_main_mean,
        "auc_5yr_sd": auc_main_sd,
        "auc_5yr_risk_mean": risk_mean,
        "auc_5yr_risk_sd": risk_sd,
        "auc_5yr_logistic_mean": log_mean,
        "auc_5yr_logistic_sd": log_sd,
    }
    return summary, fold_rows


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def modality_counts(features: Sequence[str]) -> Dict[str, int]:
    counts = {p.rstrip("_"): 0 for p in ALL_MODALITY_PREFIXES}
    counts["OTHER"] = 0
    for f in features:
        counts[modality_of(f)] = counts.get(modality_of(f), 0) + 1
    return counts


def fmt_mean_sd(mean: float, sd: float, ndigits: int = 3) -> str:
    return f"{mean:.{ndigits}f} ± {sd:.{ndigits}f}"


def make_manuscript_table(best_df: pd.DataFrame) -> pd.DataFrame:
    clinical_row = best_df[best_df["pool"] == "Clinical-only pipeline"].iloc[0]
    rows = []
    for _, r in best_df.iterrows():
        delta_c = float(r["c_index_mean"] - clinical_row["c_index_mean"])
        delta_auc = float(r["auc_5yr_mean"] - clinical_row["auc_5yr_mean"])
        rows.append(
            {
                "Model": r["pool"],
                "Algorithm": r["algorithm"],
                "alpha": r["alpha"],
                "Cox penalizer": r["penalizer"],
                "No. features": int(r["n_features"]),
                "Clinical features": int(r["n_clinical"]),
                "Omics features": int(r["n_omics"]),
                "C-index": fmt_mean_sd(float(r["c_index_mean"]), float(r["c_index_sd"])),
                "5-year AUC": fmt_mean_sd(float(r["auc_5yr_mean"]), float(r["auc_5yr_sd"])),
                "Δ C-index vs clinical": "reference" if r["pool"] == "Clinical-only pipeline" else f"{delta_c:+.3f}",
                "Δ AUC vs clinical": "reference" if r["pool"] == "Clinical-only pipeline" else f"{delta_auc:+.3f}",
            }
        )
    return pd.DataFrame(rows)


def safe_to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    try:
        return table.to_latex(index=False, escape=True, caption=caption, label=label)
    except Exception:
        return table.to_string(index=False)


def make_barplot(table: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        labels = table["pool"].str.replace(" pipeline", "", regex=False).tolist()
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, table["c_index_mean"].values.astype(float), width, label="C-index")
        ax.bar(x + width / 2, table["auc_5yr_mean"].values.astype(float), width, label="5-year AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Cross-validated performance")
        ax.set_title("Reviewer 3 pipeline-level ablation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"WARNING: could not create barplot: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewer 3 pipeline-level MB ablation.")
    parser.add_argument("--dataset-name", default="08_composite", help="Merged dataset name without .csv")
    parser.add_argument("--data-file", default=None, help="Path to merged data CSV. Defaults to ../MERGE_continuous_outer/08_composite.csv")
    parser.add_argument("--event-col", default="OS", help="Event column")
    parser.add_argument("--duration-col", default="OS.time", help="Survival duration column")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to MB/results_reviewer3_pipeline_mb_ablation")

    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS), choices=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--top-n", type=int, default=20, help="Final number of selected features per model")
    parser.add_argument("--min-mb-features", type=int, default=20, help="Fallback target if MB returns too few features")
    parser.add_argument("--mb-candidate-cap", type=int, default=50, help="Candidate cap before final top-N refinement")
    parser.add_argument("--n-perm", type=int, default=50, help="Permutation count for MI threshold")
    parser.add_argument("--dcor-max-samples", type=int, default=500, help="Max sample size for approximate distance correlation")
    parser.add_argument("--skip-dcor", action="store_true", help="Skip distance correlation in fallback ranking for speed")
    parser.add_argument("--final-ranker", default="cox", choices=["cox", "composite", "mi"], help="How to refine MB candidates to top-N")

    parser.add_argument("--penalizer-grid", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--cox-rank-penalizer", type=float, default=5.0, help="Penalizer for univariate Cox ranking")
    parser.add_argument("--scale-for-cox", action="store_true", help="Scale features before Cox fitting/evaluation")
    parser.add_argument("--auc-mode", default="logistic", choices=["logistic", "risk", "both"], help="Main 5-year AUC calculation")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--survival-years", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_dir = infer_project_dir()
    mb_dir = project_dir / "MB"
    if args.data_file:
        data_file = Path(args.data_file)
        if not data_file.is_absolute():
            data_file = (Path.cwd() / data_file).resolve()
    else:
        found = find_dataset_file(project_dir, args.dataset_name)
        if found is None:
            raise FileNotFoundError(
                f"Could not find {args.dataset_name}.csv. Pass --data-file explicitly. "
                f"Project dir inferred as: {project_dir}"
            )
        data_file = found

    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = (Path.cwd() / out_dir).resolve()
    else:
        out_dir = mb_dir / "results_reviewer3_pipeline_mb_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Reviewer 3 PIPELINE-LEVEL MB ablation")
    print("=" * 78)
    print(f"Project dir: {project_dir}")
    print(f"Data file:   {data_file}")
    print(f"Output dir:  {out_dir}")
    print(f"Algorithms:  {args.algorithms}")
    print(f"Alphas:      {args.alphas}")
    print(f"Top-N:       {args.top_n}")
    print(f"Penalizers:  {args.penalizer_grid}")
    print(f"AUC mode:    {args.auc_mode}")
    print()

    t_start = time.time()
    df = read_merged_dataset(data_file, event_col=args.event_col, duration_col=args.duration_col)
    all_features_raw = feature_columns(df, args.event_col, args.duration_col)
    all_features_raw = [f for f in all_features_raw if any(f.startswith(p) for p in ALL_MODALITY_PREFIXES)]

    if not all_features_raw:
        raise ValueError("No modality-prefixed features found. Expected CLIN_, RNA_, CNV_, MUT_, PROT_, METH_, MIRNA_.")

    # Coerce once and remove unusable features globally.
    X_all_numeric = coerce_numeric_frame(df, all_features_raw)
    X_all_numeric = drop_unusable_features(X_all_numeric, min_nonmissing=10)
    all_features = list(X_all_numeric.columns)

    clinical_features = [f for f in all_features if is_clinical_feature(f)]
    omics_features = [f for f in all_features if is_omics_feature(f)]
    combined_features = clinical_features + omics_features

    pools: Dict[str, List[str]] = {
        "Clinical-only pipeline": clinical_features,
        "Omics-only pipeline": omics_features,
        "Clinical + omics pipeline": combined_features,
    }

    print("COHORT")
    print("-" * 78)
    print(f"Rows after outcome filtering: {len(df)}")
    print(f"Observed events: {int(df[args.event_col].sum())}")
    print(f"Candidate features after numeric/constant filtering: {len(all_features)}")
    for name, feats in pools.items():
        print(f"  {name}: {len(feats)} candidate features")
    print()

    # Global Cox p-value ranking, computed once for all available candidate features.
    print("Computing global univariate Cox ranks for final top-N refinement...")
    cox_rank_t0 = time.time()
    cox_pvals = compute_global_cox_pvals(
        df=df,
        features=all_features,
        event_col=args.event_col,
        duration_col=args.duration_col,
        penalizer=args.cox_rank_penalizer,
        scale_for_cox=args.scale_for_cox,
    )
    print(f"  Done in {time.time() - cox_rank_t0:.1f}s")
    print()

    all_config_rows: List[Dict[str, object]] = []
    all_fold_rows: List[Dict[str, object]] = []
    all_selected_rows: List[Dict[str, object]] = []

    for pool_name, pool_feats in pools.items():
        if not pool_feats:
            print(f"Skipping {pool_name}: no features.")
            continue
        print("=" * 78)
        print(pool_name)
        print("=" * 78)
        print(f"Pool candidate features: {len(pool_feats)}")

        for algorithm in args.algorithms:
            for alpha in args.alphas:
                print(f"\n[{pool_name}] MB selection: {algorithm}, alpha={alpha}")
                sel_t0 = time.time()
                try:
                    sel = select_pipeline_features(
                        df=df,
                        pool_name=pool_name,
                        pool_features=pool_feats,
                        algorithm=algorithm,
                        alpha=alpha,
                        event_col=args.event_col,
                        duration_col=args.duration_col,
                        top_n=args.top_n,
                        mb_candidate_cap=args.mb_candidate_cap,
                        min_mb_features=args.min_mb_features,
                        n_perm=args.n_perm,
                        random_seed=args.random_seed,
                        cox_pvals=cox_pvals,
                        dcor_max_samples=args.dcor_max_samples,
                        skip_dcor=args.skip_dcor,
                        final_ranker=args.final_ranker,
                    )
                except Exception as e:
                    print(f"  SELECTION ERROR: {e}")
                    continue
                sel_time = time.time() - sel_t0
                mcounts = modality_counts(sel.selected_features)
                n_clin = int(mcounts.get("CLIN", 0))
                n_omics = int(sum(mcounts.get(p.rstrip("_"), 0) for p in OMICS_PREFIXES))
                print(
                    f"  MB raw={sel.mb_initial_n}, fallback_added={sel.fallback_added_n}, "
                    f"candidate={sel.mb_candidate_n}, final={len(sel.selected_features)} "
                    f"(clinical={n_clin}, omics={n_omics}), selection_time={sel_time:.1f}s"
                )

                for rank, f in enumerate(sel.selected_features, start=1):
                    all_selected_rows.append(
                        {
                            "pool": pool_name,
                            "algorithm": algorithm,
                            "alpha": alpha,
                            "rank": rank,
                            "feature": f,
                            "modality": modality_of(f),
                            "cox_p": float(cox_pvals.get(f, 1.0)),
                            "mi_score": float(sel.mi_scores.get(f, np.nan)),
                            "composite_rank": float(sel.composite_ranks.get(f, np.nan)),
                            "source": "MB" if f in sel.mb_features else "fallback",
                        }
                    )

                for penalizer in args.penalizer_grid:
                    print(f"    Evaluating Cox penalizer={penalizer} ...", end="", flush=True)
                    eval_t0 = time.time()
                    try:
                        metrics, fold_rows = evaluate_cvx_cox(
                            df=df,
                            features=sel.selected_features,
                            event_col=args.event_col,
                            duration_col=args.duration_col,
                            penalizer=float(penalizer),
                            n_folds=args.n_folds,
                            random_seed=args.random_seed,
                            survival_years=args.survival_years,
                            scale_for_cox=args.scale_for_cox,
                            auc_mode=args.auc_mode,
                        )
                    except Exception as e:
                        print(f" ERROR: {e}")
                        continue
                    eval_time = time.time() - eval_t0
                    print(
                        f" C-index={metrics['c_index_mean']:.4f} ± {metrics['c_index_sd']:.4f}; "
                        f"AUC-5yr={metrics['auc_5yr_mean']:.4f} ± {metrics['auc_5yr_sd']:.4f} "
                        f"({eval_time:.1f}s)"
                    )

                    row = {
                        "pool": pool_name,
                        "algorithm": algorithm,
                        "alpha": float(alpha),
                        "penalizer": float(penalizer),
                        "n_pool_features": int(sel.pool_n_features),
                        "mb_initial_n": int(sel.mb_initial_n),
                        "fallback_added_n": int(sel.fallback_added_n),
                        "mb_candidate_n": int(sel.mb_candidate_n),
                        "n_features": int(len(sel.selected_features)),
                        "n_clinical": n_clin,
                        "n_omics": n_omics,
                        "selection_time_sec": round(sel_time, 3),
                        "evaluation_time_sec": round(eval_time, 3),
                        **{k: float(v) for k, v in metrics.items()},
                    }
                    for mod_key, mod_n in mcounts.items():
                        row[f"n_{mod_key.lower()}"] = int(mod_n)
                    all_config_rows.append(row)

                    for fr in fold_rows:
                        fr2 = dict(fr)
                        fr2.update(
                            {
                                "pool": pool_name,
                                "algorithm": algorithm,
                                "alpha": float(alpha),
                                "penalizer": float(penalizer),
                            }
                        )
                        all_fold_rows.append(fr2)

    if not all_config_rows:
        raise RuntimeError("No successful configurations were evaluated.")

    all_df = pd.DataFrame(all_config_rows)
    fold_df = pd.DataFrame(all_fold_rows)
    selected_df = pd.DataFrame(all_selected_rows)

    sort_cols = ["pool", "c_index_mean", "auc_5yr_mean"]
    all_df = all_df.sort_values(sort_cols, ascending=[True, False, False]).reset_index(drop=True)

    # Best independently tuned result for each pool.
    best_rows = []
    for pool_name in pools:
        sub = all_df[all_df["pool"] == pool_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["c_index_mean", "auc_5yr_mean", "n_features"], ascending=[False, False, True])
        best_rows.append(sub.iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)

    # Fixed-config table: use the best Clinical+omics algorithm/alpha/penalizer across all pools.
    fixed_df = pd.DataFrame()
    if not best_df.empty and (best_df["pool"] == "Clinical + omics pipeline").any():
        multimodal_best = best_df[best_df["pool"] == "Clinical + omics pipeline"].iloc[0]
        fixed_alg = multimodal_best["algorithm"]
        fixed_alpha = float(multimodal_best["alpha"])
        fixed_pen = float(multimodal_best["penalizer"])
        fixed_df = all_df[
            (all_df["algorithm"] == fixed_alg)
            & (np.isclose(all_df["alpha"].astype(float), fixed_alpha))
            & (np.isclose(all_df["penalizer"].astype(float), fixed_pen))
        ].copy()
        fixed_df = fixed_df.sort_values("pool").reset_index(drop=True)

    # Deltas vs clinical for best-by-pool and fixed config.
    def add_deltas(tbl: pd.DataFrame) -> pd.DataFrame:
        tbl = tbl.copy()
        if tbl.empty or not (tbl["pool"] == "Clinical-only pipeline").any():
            tbl["delta_c_index_vs_clinical"] = np.nan
            tbl["delta_auc_5yr_vs_clinical"] = np.nan
            return tbl
        clin = tbl[tbl["pool"] == "Clinical-only pipeline"].iloc[0]
        tbl["delta_c_index_vs_clinical"] = tbl["c_index_mean"].astype(float) - float(clin["c_index_mean"])
        tbl["delta_auc_5yr_vs_clinical"] = tbl["auc_5yr_mean"].astype(float) - float(clin["auc_5yr_mean"])
        return tbl

    best_df = add_deltas(best_df)
    fixed_df = add_deltas(fixed_df) if not fixed_df.empty else fixed_df

    # Manuscript table defaults to independently tuned best-by-pool.
    manuscript_table = make_manuscript_table(best_df)

    # Save outputs.
    all_df.to_csv(out_dir / "reviewer3_pipeline_all_configs.csv", index=False)
    fold_df.to_csv(out_dir / "reviewer3_pipeline_fold_metrics.csv", index=False)
    best_df.to_csv(out_dir / "reviewer3_pipeline_best_by_pool.csv", index=False)
    if not fixed_df.empty:
        fixed_df.to_csv(out_dir / "reviewer3_pipeline_fixed_multimodal_config.csv", index=False)
    selected_df.to_csv(out_dir / "reviewer3_pipeline_selected_features.csv", index=False)
    manuscript_table.to_csv(out_dir / "reviewer3_pipeline_table_for_manuscript.csv", index=False)

    latex = safe_to_latex(
        manuscript_table,
        caption=(
            "Pipeline-level added-value ablation of clinical and molecular features. "
            "Feature selection was re-run separately within clinical-only, omics-only, "
            "and clinical-plus-omics candidate spaces."
        ),
        label="tab:reviewer3_pipeline_ablation",
    )
    (out_dir / "reviewer3_pipeline_table_for_manuscript.tex").write_text(latex, encoding="utf-8")

    make_barplot(best_df, out_dir / "reviewer3_pipeline_barplot.png")

    metadata = {
        "project_dir": str(project_dir),
        "data_file": str(data_file),
        "output_dir": str(out_dir),
        "dataset_name": args.dataset_name,
        "event_col": args.event_col,
        "duration_col": args.duration_col,
        "n_rows": int(len(df)),
        "n_events": int(df[args.event_col].sum()),
        "n_candidate_features": int(len(all_features)),
        "n_clinical_candidate_features": int(len(clinical_features)),
        "n_omics_candidate_features": int(len(omics_features)),
        "algorithms": args.algorithms,
        "alphas": args.alphas,
        "top_n": args.top_n,
        "min_mb_features": args.min_mb_features,
        "mb_candidate_cap": args.mb_candidate_cap,
        "n_perm": args.n_perm,
        "penalizer_grid": args.penalizer_grid,
        "cox_rank_penalizer": args.cox_rank_penalizer,
        "scale_for_cox": bool(args.scale_for_cox),
        "auc_mode": args.auc_mode,
        "n_folds": args.n_folds,
        "random_seed": args.random_seed,
        "survival_years": args.survival_years,
        "runtime_sec": round(time.time() - t_start, 3),
    }
    (out_dir / "reviewer3_pipeline_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Human-readable report.
    report_lines: List[str] = []
    report_lines += [
        "REVIEWER 3 PIPELINE-LEVEL MB ABLATION ANALYSIS",
        "=" * 78,
        f"Project dir: {project_dir}",
        f"Data file: {data_file}",
        f"Output dir: {out_dir}",
        "",
        "PURPOSE",
        "-" * 78,
        "This analysis re-runs feature selection within three candidate spaces:",
        "  1. clinical-only features",
        "  2. omics-only features",
        "  3. clinical + omics features",
        "rather than only decomposing the already selected final 20-feature signature.",
        "",
        "SETTINGS",
        "-" * 78,
        f"Dataset name: {args.dataset_name}",
        f"Event column: {args.event_col}",
        f"Duration column: {args.duration_col}",
        f"Algorithms: {args.algorithms}",
        f"Alphas: {args.alphas}",
        f"Top-N final selected features: {args.top_n}",
        f"Minimum MB/fallback features: {args.min_mb_features}",
        f"MB candidate cap: {args.mb_candidate_cap}",
        f"Final ranker: {args.final_ranker}",
        f"Cox penalizer grid: {args.penalizer_grid}",
        f"AUC mode: {args.auc_mode}",
        f"CV folds: {args.n_folds}",
        f"Random seed: {args.random_seed}",
        f"Scale features for Cox: {args.scale_for_cox}",
        "",
        "COHORT",
        "-" * 78,
        f"Rows after OS/OS.time filtering: {len(df)}",
        f"Observed events: {int(df[args.event_col].sum())}",
        f"Candidate features after numeric/constant filtering: {len(all_features)}",
        f"Clinical candidate features: {len(clinical_features)}",
        f"Omics candidate features: {len(omics_features)}",
        "",
        "BEST-BY-POOL RESULTS",
        "-" * 78,
        best_df[[
            "pool",
            "algorithm",
            "alpha",
            "penalizer",
            "n_features",
            "n_clinical",
            "n_omics",
            "c_index_mean",
            "c_index_sd",
            "auc_5yr_mean",
            "auc_5yr_sd",
            "delta_c_index_vs_clinical",
            "delta_auc_5yr_vs_clinical",
        ]].to_string(index=False),
        "",
        "TABLE READY FOR MANUSCRIPT",
        "-" * 78,
        manuscript_table.to_string(index=False),
    ]

    if not fixed_df.empty:
        report_lines += [
            "",
            "FIXED MULTIMODAL-CONFIG RESULTS",
            "-" * 78,
            "This table uses the best clinical+omics algorithm/alpha/penalizer and applies that same setting to all pools.",
            fixed_df[[
                "pool",
                "algorithm",
                "alpha",
                "penalizer",
                "n_features",
                "n_clinical",
                "n_omics",
                "c_index_mean",
                "c_index_sd",
                "auc_5yr_mean",
                "auc_5yr_sd",
                "delta_c_index_vs_clinical",
                "delta_auc_5yr_vs_clinical",
            ]].to_string(index=False),
        ]

    # Interpretation line from best-by-pool if possible.
    try:
        clin = best_df[best_df["pool"] == "Clinical-only pipeline"].iloc[0]
        multi = best_df[best_df["pool"] == "Clinical + omics pipeline"].iloc[0]
        dc = float(multi["c_index_mean"] - clin["c_index_mean"])
        da = float(multi["auc_5yr_mean"] - clin["auc_5yr_mean"])
        interp = (
            f"In the pipeline-level ablation, the best clinical-plus-omics pipeline changed "
            f"the C-index by {dc:+.3f} and the 5-year AUC by {da:+.3f} relative to the "
            f"best clinical-only pipeline."
        )
        report_lines += ["", "INTERPRETATION TEMPLATE", "-" * 78, interp]
    except Exception:
        pass

    report_lines += [
        "",
        "FILES WRITTEN",
        "-" * 78,
        "reviewer3_pipeline_all_configs.csv",
        "reviewer3_pipeline_fold_metrics.csv",
        "reviewer3_pipeline_best_by_pool.csv",
        "reviewer3_pipeline_fixed_multimodal_config.csv",
        "reviewer3_pipeline_selected_features.csv",
        "reviewer3_pipeline_table_for_manuscript.csv",
        "reviewer3_pipeline_table_for_manuscript.tex",
        "reviewer3_pipeline_report.txt",
        "reviewer3_pipeline_barplot.png",
        "reviewer3_pipeline_run_metadata.json",
        "",
        f"Done in {time.time() - t_start:.1f}s",
    ]
    report_text = "\n".join(report_lines)
    (out_dir / "reviewer3_pipeline_report.txt").write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
