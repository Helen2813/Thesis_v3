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
N_TOP_DISPLAY = 20
PENALIZER_GRID = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]


def project_dir_from_here() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "MB":
        return here.parents[1]
    if (here / "MERGE_continuous_outer").exists():
        return here
    cwd = Path.cwd().resolve()
    if cwd.name == "MB":
        return cwd.parent
    if (cwd / "MERGE_continuous_outer").exists():
        return cwd
    raise FileNotFoundError("Run from 01_Causal_feature_extraction or 01_Causal_feature_extraction/MB.")


def read_selected_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Selected features file not found: {path}")
    df = pd.read_csv(path)
    candidates = [c for c in df.columns if "feature" in c.lower()]
    if candidates:
        vals = df[candidates[0]].dropna().astype(str).tolist()
    elif df.shape[1] == 1:
        vals = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        vals = df.iloc[:, -1].dropna().astype(str).tolist()
    bad = {EVENT_COL, TIME_COL, "patient_id", "case_id", "sample_id"}
    features = []
    for f in vals:
        f = f.strip()
        if f and f not in bad and f not in features:
            features.append(f)
    return features


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
    prefixes = ["CLIN_", "RNA_", "CNV_", "MUT_", "PROT_", "METH_", "MIRNA_"]
    out = feature
    for p in prefixes:
        if out.startswith(p):
            out = out[len(p):]
            break
    out = out.replace(".diagnoses", "").replace(".demographic", "").replace(".treatments", "")
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


def prepare_data(data_path: Path, features: list[str]):
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
        if work[f].isna().any():
            work[f] = work[f].fillna(work[f].median())

    kept = []
    dropped = []
    for f in features:
        if work[f].nunique(dropna=True) <= 1:
            dropped.append(f)
        else:
            kept.append(f)

    work = work[[TIME_COL, EVENT_COL] + kept].dropna().copy()
    return work, kept, dropped


def fit_cox(work: pd.DataFrame, features: list[str]):
    last_error = None
    for pen in PENALIZER_GRID:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(work[[TIME_COL, EVENT_COL] + features], duration_col=TIME_COL, event_col=EVENT_COL)
            return cph, pen, None
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"Cox model failed for all penalizers. Last error: {last_error}")


def make_results(cph, features: list[str]) -> pd.DataFrame:
    s = cph.summary.reset_index().rename(columns={"covariate": "feature", "index": "feature"})
    if "feature" not in s.columns:
        s = s.rename(columns={s.columns[0]: "feature"})

    cols = {
        "coef": "coef",
        "exp(coef)": "hazard_ratio",
        "se(coef)": "se",
        "p": "p_value",
        "exp(coef) lower 95%": "hr_ci_lower",
        "exp(coef) upper 95%": "hr_ci_upper",
    }

    out = pd.DataFrame()
    out["feature"] = s["feature"]
    for src, dst in cols.items():
        out[dst] = s[src] if src in s.columns else np.nan

    if out["hr_ci_lower"].isna().all() or out["hr_ci_upper"].isna().all():
        ci = cph.confidence_intervals_.reset_index()
        ci_cols = [c for c in ci.columns if c != ci.columns[0]]
        if len(ci_cols) >= 2:
            lower = np.exp(ci[ci_cols[0]])
            upper = np.exp(ci[ci_cols[1]])
            out["hr_ci_lower"] = lower.values
            out["hr_ci_upper"] = upper.values

    out["modality"] = out["feature"].map(modality)
    out["display_feature"] = out["feature"].map(display_name)
    out["abs_log_hr"] = out["coef"].abs()
    out = out.sort_values(["p_value", "abs_log_hr"], ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def write_latex_table(results: pd.DataFrame, path: Path):
    rows = []
    for _, r in results.head(N_TOP_DISPLAY).iterrows():
        hr = f"{r['hazard_ratio']:.3f}"
        ci = f"{r['hr_ci_lower']:.3f}--{r['hr_ci_upper']:.3f}"
        rows.append(
            f"{latex_escape(r['display_feature'])} & {latex_escape(r['modality'])} & "
            f"${hr}$ & $({ci})$ & ${format_p(r['p_value'])}$ \\\\"
        )

    text = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Multivariate Cox regression for the final Markov Blanket-derived 20-feature signature. Hazard ratios are estimated with all retained features included in the same model.}
\label{tab:multivariate_cox_signature}
\begin{tabular}{p{0.43\textwidth}lccc}
\hline
Feature & Modality & HR & 95\% CI & p-value \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
    path.write_text(text, encoding="utf-8")


def write_forest_plot(results: pd.DataFrame, path: Path):
    if plt is None:
        return
    plot_df = results.head(N_TOP_DISPLAY).copy()
    plot_df = plot_df.iloc[::-1]
    y = np.arange(len(plot_df))

    fig_h = max(6, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    hr = plot_df["hazard_ratio"].values
    lo = plot_df["hr_ci_lower"].values
    hi = plot_df["hr_ci_upper"].values
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_feature"].astype(str).values, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio, log scale")
    ax.set_title("Multivariate Cox regression: final MB-derived signature")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    project_dir = project_dir_from_here()
    mb_dir = project_dir / "MB"
    data_path = project_dir / "MERGE_continuous_outer" / f"{DATASET_NAME}.csv"
    selected_path = mb_dir / "results_final" / "selected_features" / f"{DATASET_NAME}_features.csv"
    out_dir = mb_dir / "results_reviewer2_multivariate_cox"
    out_dir.mkdir(parents=True, exist_ok=True)

    features = read_selected_features(selected_path)
    work, kept_features, dropped_features = prepare_data(data_path, features)
    cph, used_penalizer, fit_warning = fit_cox(work, kept_features)
    results = make_results(cph, kept_features)

    full_csv = out_dir / "reviewer2_multivariate_cox_results.csv"
    table_csv = out_dir / "reviewer2_multivariate_cox_table_for_manuscript.csv"
    tex_path = out_dir / "reviewer2_multivariate_cox_table_for_manuscript.tex"
    report_path = out_dir / "reviewer2_multivariate_cox_report.txt"
    ph_path = out_dir / "reviewer2_multivariate_cox_ph_test.csv"
    plot_path = out_dir / "reviewer2_multivariate_cox_forestplot.png"
    metadata_path = out_dir / "reviewer2_multivariate_cox_metadata.json"

    results.to_csv(full_csv, index=False)
    manuscript = results[["rank", "display_feature", "feature", "modality", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]].copy()
    manuscript["HR (95% CI)"] = manuscript.apply(
        lambda r: f"{r['hazard_ratio']:.3f} ({r['hr_ci_lower']:.3f}-{r['hr_ci_upper']:.3f})",
        axis=1,
    )
    manuscript["p-value"] = manuscript["p_value"].map(format_p)
    manuscript.to_csv(table_csv, index=False)
    write_latex_table(results, tex_path)
    write_forest_plot(results, plot_path)

    try:
        ph = proportional_hazard_test(cph, work[[TIME_COL, EVENT_COL] + kept_features], time_transform="rank")
        ph.summary.reset_index().rename(columns={"index": "feature"}).to_csv(ph_path, index=False)
        ph_status = f"PH test written: {ph_path.name}"
    except Exception as exc:
        ph_status = f"PH test not written: {exc}"

    significant = int((results["p_value"] < 0.05).sum())
    report = []
    report.append("REVIEWER 2 MULTIVARIATE COX REGRESSION")
    report.append("=" * 80)
    report.append(f"Data file: {data_path}")
    report.append(f"Selected features file: {selected_path}")
    report.append(f"Samples used: {len(work)}")
    report.append(f"Observed events: {int(work[EVENT_COL].sum())}")
    report.append(f"Requested features: {len(features)}")
    report.append(f"Features used: {len(kept_features)}")
    report.append(f"Dropped zero-variance features: {len(dropped_features)}")
    report.append(f"Cox penalizer used: {used_penalizer}")
    report.append(f"Concordance index, full fit: {cph.concordance_index_:.4f}")
    report.append(f"Log-likelihood: {cph.log_likelihood_:.4f}")
    if hasattr(cph, "AIC_partial_"):
        report.append(f"Partial AIC: {cph.AIC_partial_:.4f}")
    report.append(f"Features with p < 0.05: {significant}")
    report.append(ph_status)
    report.append("")
    report.append("TOP RESULTS")
    report.append("-" * 80)
    top_cols = ["display_feature", "modality", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]
    report.append(results[top_cols].head(N_TOP_DISPLAY).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    report.append("")
    report.append("INTERPRETATION TEMPLATE")
    report.append("-" * 80)
    report.append(
        f"A multivariate Cox regression model was fitted using the final Markov Blanket-derived signature "
        f"with all retained features included simultaneously (n={len(work)}, events={int(work[EVENT_COL].sum())}). "
        f"The model achieved an apparent concordance index of {cph.concordance_index_:.3f}. "
        f"{significant} retained features had p < 0.05 in the multivariate model."
    )

    report_path.write_text("\n".join(report), encoding="utf-8")

    metadata = {
        "data_path": str(data_path),
        "selected_features_path": str(selected_path),
        "output_dir": str(out_dir),
        "samples_used": int(len(work)),
        "events": int(work[EVENT_COL].sum()),
        "requested_features": int(len(features)),
        "features_used": int(len(kept_features)),
        "dropped_zero_variance_features": dropped_features,
        "cox_penalizer_used": float(used_penalizer),
        "concordance_index_full_fit": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "significant_p_lt_0_05": int(significant),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Reviewer 2 multivariate Cox regression")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Selected features: {selected_path}")
    print(f"Output: {out_dir}")
    print(f"Samples used: {len(work)} | Events: {int(work[EVENT_COL].sum())} | Features: {len(kept_features)}")
    print(f"Cox penalizer used: {used_penalizer}")
    print(f"Concordance index, full fit: {cph.concordance_index_:.4f}")
    print(f"Features with p < 0.05: {significant}")
    print("")
    print(results[["rank", "display_feature", "modality", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]].head(N_TOP_DISPLAY).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("")
    print("Files written:")
    for p in [full_csv, table_csv, tex_path, report_path, ph_path, plot_path, metadata_path]:
        if p.exists():
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
