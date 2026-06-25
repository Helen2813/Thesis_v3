from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
except ImportError as exc:
    raise SystemExit("Missing dependency: lifelines. Install with: pip install lifelines") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DATASET_NAME = "08_composite"
EVENT_COL = "OS"
TIME_COL = "OS.time"
MIN_LEVEL_N = 12
MIN_LEVEL_EVENTS = 3
PENALIZERS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
EXCLUDE_FOR_INFERENCE = [
    "tissue_or_organ_of_origin",
    "ajcc_staging_system_edition",
]


def project_dir_from_here() -> Path:
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for base in [here.parent, cwd]:
        if base.name == "MB":
            return base.parent
        if (base / "MERGE_continuous_outer").exists():
            return base
    raise FileNotFoundError("Run from 01_Causal_feature_extraction or 01_Causal_feature_extraction/MB.")


def read_selected_features(path: Path) -> list[str]:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if "feature" in c.lower()]
    vals = df[cols[0]].dropna().astype(str).tolist() if cols else df.iloc[:, -1].dropna().astype(str).tolist()
    bad = {EVENT_COL, TIME_COL, "patient_id", "case_id", "sample_id"}
    out = []
    for v in vals:
        v = v.strip()
        if v and v not in bad and v not in out:
            out.append(v)
    return out


def modality(feature: str) -> str:
    if feature.startswith("CLIN_"):
        return "Clinical"
    if feature.startswith("RNA_"):
        return "RNA"
    if feature.startswith("CNV_"):
        return "CNV"
    if feature.startswith("MUT_"):
        return "Mutation"
    if feature.startswith("PROT_"):
        return "Protein"
    if feature.startswith("METH_"):
        return "Methylation"
    if feature.startswith("MIRNA_"):
        return "miRNA"
    return "Other"


def display_name(feature: str) -> str:
    out = feature
    for p in ["CLIN_", "RNA_", "CNV_", "MUT_", "PROT_", "METH_", "MIRNA_"]:
        if out.startswith(p):
            out = out[len(p):]
            break
    out = out.replace(".diagnoses", "")
    out = out.replace(".demographic", "")
    out = out.replace(".treatments", "")
    out = out.replace(".samples", "")
    return out


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def format_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def is_binary(s: pd.Series) -> bool:
    vals = set(pd.Series(s.dropna().unique()).round(10).tolist())
    return vals.issubset({0, 1}) and len(vals) <= 2


def load_data(data_path: Path, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(data_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError("Selected features missing from data:\n" + "\n".join(missing))
    cols = [TIME_COL, EVENT_COL] + features
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[TIME_COL, EVENT_COL])
    work = work[(work[TIME_COL] > 0) & (work[EVENT_COL].isin([0, 1]))].copy()
    for f in features:
        work[f] = work[f].fillna(work[f].median())
    kept = [f for f in features if work[f].nunique(dropna=True) > 1]
    work = work[[TIME_COL, EVENT_COL] + kept].dropna().copy()
    return work, kept


def stable_feature_filter(work: pd.DataFrame, features: list[str]) -> tuple[list[str], pd.DataFrame]:
    rows = []
    kept = []
    for f in features:
        reason = "kept"
        if any(x in f for x in EXCLUDE_FOR_INFERENCE):
            reason = "excluded_admin_or_site_indicator"
        elif is_binary(work[f]):
            x = work[f].astype(int)
            y = work[EVENT_COL].astype(int)
            n1 = int((x == 1).sum())
            n0 = int((x == 0).sum())
            e1 = int(y[x == 1].sum())
            e0 = int(y[x == 0].sum())
            ne1 = n1 - e1
            ne0 = n0 - e0
            if min(n1, n0) < MIN_LEVEL_N or min(e1, e0) < MIN_LEVEL_EVENTS or min(ne1, ne0) < MIN_LEVEL_EVENTS:
                reason = "excluded_sparse_binary_level"
        if reason == "kept":
            kept.append(f)
        rows.append({"feature": f, "display_feature": display_name(f), "modality": modality(f), "status": reason})
    return kept, pd.DataFrame(rows)


def standardize_continuous(work: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = work[[TIME_COL, EVENT_COL] + features].copy()
    standardized = []
    for f in features:
        if not is_binary(out[f]):
            sd = out[f].std(ddof=0)
            if sd and np.isfinite(sd) and sd > 0:
                out[f] = (out[f] - out[f].mean()) / sd
                standardized.append(f)
    return out, standardized


def fit_model(work: pd.DataFrame, features: list[str]):
    best = None
    records = []
    for pen in PENALIZERS:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(work[[TIME_COL, EVENT_COL] + features], TIME_COL, EVENT_COL)
            s = cph.summary.copy()
            hr = s["exp(coef)"].replace([np.inf, -np.inf], np.nan)
            extreme = int(((hr > 1e4) | (hr < 1e-4)).sum())
            rec = {"penalizer": pen, "c_index": float(cph.concordance_index_), "extreme_hr_count": extreme, "model": cph}
            records.append(rec)
            if extreme == 0 and (best is None or pen < best["penalizer"]):
                best = rec
        except Exception:
            continue
    if best is None and records:
        records = sorted(records, key=lambda r: (r["extreme_hr_count"], -r["c_index"], r["penalizer"]))
        best = records[0]
    if best is None:
        raise RuntimeError("Cox model failed for all penalizers.")
    return best["model"], float(best["penalizer"]), pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in records])


def results_table(cph, features: list[str], standardized: list[str]) -> pd.DataFrame:
    s = cph.summary.reset_index().rename(columns={"covariate": "feature", "index": "feature"})
    if "feature" not in s.columns:
        s = s.rename(columns={s.columns[0]: "feature"})
    out = pd.DataFrame()
    out["feature"] = s["feature"]
    out["display_feature"] = out["feature"].map(display_name)
    out["modality"] = out["feature"].map(modality)
    out["hazard_ratio"] = s.get("exp(coef)", np.nan)
    out["hr_ci_lower"] = s.get("exp(coef) lower 95%", np.nan)
    out["hr_ci_upper"] = s.get("exp(coef) upper 95%", np.nan)
    out["p_value"] = s.get("p", np.nan)
    out["scale"] = np.where(out["feature"].isin(standardized), "per 1 SD", "category vs reference")
    out["abs_log_hr"] = s.get("coef", 0).abs()
    out = out.sort_values(["p_value", "abs_log_hr"], ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def write_latex(results: pd.DataFrame, path: Path):
    rows = []
    for _, r in results.iterrows():
        rows.append(
            f"{latex_escape(r['display_feature'])} & {latex_escape(r['modality'])} & {latex_escape(r['scale'])} & "
            f"${r['hazard_ratio']:.3f}$ & $({r['hr_ci_lower']:.3f}--{r['hr_ci_upper']:.3f})$ & ${format_p(r['p_value'])}$ \\"
        )
    text = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Multivariate Cox regression for the interpretable components of the final Markov Blanket-derived signature. Continuous variables are standardized, so their hazard ratios are reported per one standard deviation. Sparse administrative or site indicators were excluded from this inferential table to reduce separation effects.}
\label{tab:multivariate_cox_signature}
\begin{tabular}{p{0.34\textwidth}llccc}
\hline
Feature & Modality & Scale & HR & 95\% CI & p-value \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
    path.write_text(text, encoding="utf-8")


def plot_forest(results: pd.DataFrame, path: Path):
    if plt is None or results.empty:
        return
    plot_df = results.iloc[::-1].copy()
    y = np.arange(len(plot_df))
    fig_h = max(4, 0.38 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    hr = plot_df["hazard_ratio"].astype(float).values
    lo = plot_df["hr_ci_lower"].astype(float).values
    hi = plot_df["hr_ci_upper"].astype(float).values
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_feature"].astype(str).values, fontsize=8)
    ax.set_xlabel("Hazard ratio, log scale")
    ax.set_title("Multivariate Cox regression")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    project_dir = project_dir_from_here()
    mb_dir = project_dir / "MB"
    data_path = project_dir / "MERGE_continuous_outer" / f"{DATASET_NAME}.csv"
    selected_path = mb_dir / "results_final" / "selected_features" / f"{DATASET_NAME}_features.csv"
    out_dir = mb_dir / "results_reviewer2_multivariate_cox_stable"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = read_selected_features(selected_path)
    work, kept = load_data(data_path, selected)
    inference_features, feature_status = stable_feature_filter(work, kept)
    model_df, standardized = standardize_continuous(work, inference_features)
    cph, pen, pen_summary = fit_model(model_df, inference_features)
    results = results_table(cph, inference_features, standardized)

    results.to_csv(out_dir / "reviewer2_multivariate_cox_stable_results.csv", index=False)
    feature_status.to_csv(out_dir / "reviewer2_multivariate_cox_feature_status.csv", index=False)
    pen_summary.to_csv(out_dir / "reviewer2_multivariate_cox_penalizer_summary.csv", index=False)
    write_latex(results, out_dir / "reviewer2_multivariate_cox_stable_table_for_manuscript.tex")
    plot_forest(results, out_dir / "reviewer2_multivariate_cox_stable_forestplot.png")

    try:
        ph = proportional_hazard_test(cph, model_df[[TIME_COL, EVENT_COL] + inference_features], time_transform="rank")
        ph.summary.reset_index().rename(columns={"index": "feature"}).to_csv(out_dir / "reviewer2_multivariate_cox_stable_ph_test.csv", index=False)
    except Exception as exc:
        (out_dir / "reviewer2_multivariate_cox_stable_ph_test_error.txt").write_text(str(exc), encoding="utf-8")

    report = []
    report.append("REVIEWER 2 STABLE MULTIVARIATE COX REGRESSION")
    report.append("=" * 80)
    report.append(f"Data file: {data_path}")
    report.append(f"Selected features file: {selected_path}")
    report.append(f"Samples used: {len(model_df)}")
    report.append(f"Observed events: {int(model_df[EVENT_COL].sum())}")
    report.append(f"Original signature features: {len(selected)}")
    report.append(f"Features used in inferential model: {len(inference_features)}")
    report.append(f"Continuous standardized features: {len(standardized)}")
    report.append(f"Cox penalizer used: {pen}")
    report.append(f"Concordance index, full fit: {cph.concordance_index_:.4f}")
    report.append(f"Features with p < 0.05: {int((results['p_value'] < 0.05).sum())}")
    report.append("")
    report.append("FEATURE STATUS")
    report.append("-" * 80)
    report.append(feature_status.to_string(index=False))
    report.append("")
    report.append("RESULTS")
    report.append("-" * 80)
    report.append(results[["display_feature", "modality", "scale", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    (out_dir / "reviewer2_multivariate_cox_stable_report.txt").write_text("\n".join(report), encoding="utf-8")

    metadata = {
        "data_path": str(data_path),
        "selected_features_path": str(selected_path),
        "samples_used": int(len(model_df)),
        "events": int(model_df[EVENT_COL].sum()),
        "original_signature_features": int(len(selected)),
        "features_used_in_inferential_model": int(len(inference_features)),
        "standardized_features": standardized,
        "cox_penalizer_used": pen,
        "concordance_index_full_fit": float(cph.concordance_index_),
    }
    (out_dir / "reviewer2_multivariate_cox_stable_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Reviewer 2 stable multivariate Cox regression")
    print("=" * 80)
    print(f"Samples used: {len(model_df)} | Events: {int(model_df[EVENT_COL].sum())}")
    print(f"Original signature features: {len(selected)} | Inferential model features: {len(inference_features)}")
    print(f"Cox penalizer used: {pen}")
    print(f"Concordance index, full fit: {cph.concordance_index_:.4f}")
    print(results[["rank", "display_feature", "modality", "scale", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nFiles written to:", out_dir)


if __name__ == "__main__":
    main()
