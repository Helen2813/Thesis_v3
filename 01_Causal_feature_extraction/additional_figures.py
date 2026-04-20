"""
Publication figures for Paper 1:
"A Markov Blanket-Based Framework for Causal Feature Selection
in Multimodal Breast Cancer Data"

Data source:
  Thesis_v3/02_ITE/01_preprocessing/output/causal_features_dataset.csv
  - 1,076 patients, 23 columns including OS, OS.time, 20 causal features
  - same file used by Paper 2 treatment preprocessing notebook

Generates:
  fig1_cohort_outcomes.png      -- outcome + event-rate distribution
  fig2_feature_modalities.png   -- modality composition + HR forest plot
  fig3_km_curves.png            -- Kaplan-Meier, cross-validated Cox risk score
  fig4_dimensionality.png       -- feature-count reduction waterfall
  go_genes_tcga_core.txt        -- gene symbols for GO/KEGG (TCGA molecular only)
  go_genes_metabric.txt         -- gene symbols for GO/KEGG (METABRIC model)
  table_cohort.tex              -- LaTeX Table 1

Install:
  pip install lifelines matplotlib seaborn scikit-learn pandas numpy
"""

import os, sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# PATHS — configure here if layout differs
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(os.path.abspath(__file__)).parent

BASE_DIR = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]
     if (p / "02_ITE" / "01_preprocessing" / "output"
         / "causal_features_dataset.csv").exists()),
    None,
)
if BASE_DIR is None:
    sys.exit(
        "ERROR: cannot locate causal_features_dataset.csv\n"
        "Expected at: <Thesis_v3>/02_ITE/01_preprocessing/output/\n"
        "This file is produced by the treatment preprocessing notebook."
    )

# Features file for reference (Paper 1 ground truth)
FEATURES_TXT = (BASE_DIR / "01_Causal_feature_extraction" / "MB"
                / "results_fallback_experiment" / "best_strategy_features.txt")

CAUSAL_CSV  = BASE_DIR / "02_ITE" / "01_preprocessing" / "output" / "causal_features_dataset.csv"
OUTPUT_DIR  = SCRIPT_DIR / "paper1_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# JOURNAL STYLE (Springer Nature single/double column)
# ---------------------------------------------------------------------------
FIG_SINGLE = 3.5
FIG_DOUBLE = 7.25
DPI        = 300

PALETTE = {
    "clinical":    "#2166AC",
    "protein":     "#D6604D",
    "methylation": "#E08214",
    "rna":         "#1A7741",
    "grey":        "#878787",
    "light":       "#F0F4F8",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       DPI,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.6,
    "xtick.major.width":0.6,
    "ytick.major.width":0.6,
})

# ---------------------------------------------------------------------------
# CAUSAL CORE DEFINITION (from paper Section 4.4 / Table 4)
# ---------------------------------------------------------------------------
# Exact 20 features from S34_cox_pen5_top20 (best_strategy_features.txt)
# C-index=0.8085, AUC-5yr=0.8676
CAUSAL_CORE = {
    "CLIN_treatment_or_therapy.treatments.diagnoses_['not reported', 'not reported']": "Clinical",
    "CLIN_ajcc_pathologic_stage.diagnoses_Stage IV":                     "Clinical",
    "CLIN_ajcc_pathologic_m.diagnoses_M1":                               "Clinical",
    "CLIN_ajcc_pathologic_n.diagnoses_N1b":                              "Clinical",
    "CLIN_ajcc_staging_system_edition.diagnoses_5th":                    "Clinical",
    "CLIN_ajcc_staging_system_edition.diagnoses_6th":                    "Clinical",
    "CLIN_age_at_index.demographic":                                     "Clinical",
    "CLIN_treatment_or_therapy.treatments.diagnoses_['yes', 'yes']":     "Clinical",
    "CLIN_ajcc_pathologic_n.diagnoses_N0 (i-)":                          "Clinical",
    "CLIN_ajcc_pathologic_n.diagnoses_NX":                               "Clinical",
    "CLIN_ajcc_pathologic_t.diagnoses_T4b":                              "Clinical",
    "CLIN_ajcc_staging_system_edition.diagnoses_4th":                    "Clinical",
    "CLIN_ajcc_pathologic_stage.diagnoses_Stage III":                    "Clinical",
    "CLIN_tissue_or_organ_of_origin.diagnoses_Lower-inner quadrant of breast": "Clinical",
    "CLIN_tissue_or_organ_of_origin.diagnoses_Breast, NOS":              "Clinical",
    "PROT_4EBP1":                                                        "Protein",
    "PROT_ZAP-70":                                                       "Protein",
    "METH_cg00101629":                                                   "Methylation",
    "METH_cg19851563":                                                   "Methylation",
    "RNA_ENSG00000264589.4":                                             "RNA",
}

LABELS = {
    "CLIN_treatment_or_therapy.treatments.diagnoses_['not reported', 'not reported']": "Treatment not reported",
    "CLIN_ajcc_pathologic_stage.diagnoses_Stage IV":  "Stage IV",
    "CLIN_ajcc_pathologic_m.diagnoses_M1":            "M1 (distant metastasis)",
    "CLIN_ajcc_pathologic_n.diagnoses_N1b":           "N1b (lymph node)",
    "CLIN_ajcc_staging_system_edition.diagnoses_5th": "Staging edition 5th",
    "CLIN_ajcc_staging_system_edition.diagnoses_6th": "Staging edition 6th",
    "CLIN_age_at_index.demographic":                  "Age at diagnosis",
    "CLIN_treatment_or_therapy.treatments.diagnoses_['yes', 'yes']": "Treatment received",
    "CLIN_ajcc_pathologic_n.diagnoses_N0 (i-)":       "N0(i-)",
    "CLIN_ajcc_pathologic_n.diagnoses_NX":            "NX (node not assessed)",
    "CLIN_ajcc_pathologic_t.diagnoses_T4b":           "T4b (chest wall / skin)",
    "CLIN_ajcc_staging_system_edition.diagnoses_4th": "Staging edition 4th",
    "CLIN_ajcc_pathologic_stage.diagnoses_Stage III": "Stage III",
    "CLIN_tissue_or_organ_of_origin.diagnoses_Lower-inner quadrant of breast": "Lower-inner quadrant",
    "CLIN_tissue_or_organ_of_origin.diagnoses_Breast, NOS": "Breast, NOS",
    "PROT_4EBP1":                                     "4EBP1 (mTOR effector)",
    "PROT_ZAP-70":                                    "ZAP-70 (immune kinase)",
    "METH_cg00101629":                                "cg00101629 (methylation)",
    "METH_cg19851563":                                "cg19851563 (methylation)",
    "RNA_ENSG00000264589.4":                          "ENSG00000264589.4 (lncRNA)",
}

MOD_COLOR = {
    "Clinical":    PALETTE["clinical"],
    "Protein":     PALETTE["protein"],
    "Methylation": PALETTE["methylation"],
    "RNA":         PALETTE["rna"],
}

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
TCGA_NA = ["'--","--","NA","Not Reported","not reported",
           "[Not Available]","[Unknown]","[Not Applicable]"]

df = pd.read_csv(CAUSAL_CSV, na_values=TCGA_NA, low_memory=False)
if "Unnamed: 0" in df.columns:
    df.rename(columns={"Unnamed: 0": "patient_id"}, inplace=True)

required = {"OS", "OS.time"}
missing  = required - set(df.columns)
if missing:
    sys.exit(f"ERROR: required columns {missing} not found in {CAUSAL_CSV.name}")

feat_cols = [f for f in CAUSAL_CORE if f in df.columns]
if len(feat_cols) < 10:
    sys.exit(f"ERROR: only {len(feat_cols)} causal features found — check column names")

print(f"Dataset   : {df.shape[0]} patients, {df.shape[1]} columns")
print(f"Events    : {int(df['OS'].sum())} ({df['OS'].mean()*100:.1f}%)")
print(f"Features  : {len(feat_cols)} / {len(CAUSAL_CORE)} causal core found in data")

# ---------------------------------------------------------------------------
# FIG 1 — COHORT OUTCOME DISTRIBUTION
# ---------------------------------------------------------------------------

def fig1_cohort_outcomes():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 2.4))

    # Panel A: event vs censored count — from actual data
    event_counts = df["OS"].value_counts().sort_index()
    n_censored   = int(event_counts.get(0, 0))
    n_events     = int(event_counts.get(1, 0))
    labels_a     = ["Censored", "Event\n(deceased)"]
    vals_a       = [n_censored, n_events]
    colors_a     = [PALETTE["clinical"], PALETTE["protein"]]

    bars = axes[0].bar(labels_a, vals_a, color=colors_a,
                       width=0.45, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals_a):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5, str(v),
                     ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Patients (n)")
    axes[0].set_ylim(0, max(vals_a) * 1.18)
    axes[0].set_title(
        f"A  Outcome distribution\n"
        f"TCGA-BRCA (n={len(df)}, event rate={n_events/len(df)*100:.1f}%)"
    )

    # Panel B: OS.time distribution by event status — from actual data
    for ev, color, label in [(0, PALETTE["clinical"], "Censored"),
                              (1, PALETTE["protein"],  "Event")]:
        subset = df[df["OS"] == ev]["OS.time"].dropna()
        axes[1].hist(subset, bins=30, alpha=0.65, color=color,
                     label=f"{label} (n={len(subset)})", density=False)
    axes[1].set_xlabel("Overall survival time (days)")
    axes[1].set_ylabel("Patients (n)")
    axes[1].set_title("B  Survival time distribution")
    axes[1].legend(framealpha=0.7)

    plt.tight_layout(pad=0.8)
    out = OUTPUT_DIR / "fig1_cohort_outcomes.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# FIG 2 — FEATURE MODALITIES + COX HR FOREST PLOT
# ---------------------------------------------------------------------------

def fig2_feature_modalities():
    try:
        from lifelines import CoxPHFitter
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("lifelines not available — skipping fig2")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 3.8))

    # Panel A: pie chart of modality composition — actual counts from CAUSAL_CORE
    mod_counts = pd.Series(CAUSAL_CORE).value_counts()
    colors_pie = [MOD_COLOR[m] for m in mod_counts.index]
    total      = mod_counts.sum()

    def autopct_counts(pct):
        return str(int(round(pct * total / 100.0)))

    wedges, texts, autotexts = axes[0].pie(
        mod_counts.values,
        labels=mod_counts.index,
        colors=colors_pie,
        autopct=autopct_counts,
        startangle=90,
        textprops={"fontsize": 7},
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set_fontsize(7)
    axes[0].set_title(
        f"A  Causal core composition\n(n={total} features across {len(mod_counts)} modalities)"
    )

    # Panel B: Cox hazard ratios from penalized CoxPH on actual data
    sub = df[feat_cols + ["OS.time", "OS"]].copy()
    sub[feat_cols] = sub[feat_cols].fillna(0)

    # Standardise continuous features; leave binary (0/1) untouched
    binary = [f for f in feat_cols
              if set(sub[f].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    continuous = [f for f in feat_cols if f not in binary]

    if continuous:
        scaler = StandardScaler()
        sub[continuous] = scaler.fit_transform(sub[continuous])

    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(sub, duration_col="OS.time", event_col="OS")
    except Exception as e:
        print(f"  CoxPH failed: {e} — skipping HR panel")
        axes[1].axis("off")
        plt.tight_layout(pad=0.8)
        out = OUTPUT_DIR / "fig2_feature_modalities.png"
        fig.savefig(out, bbox_inches="tight", dpi=DPI)
        plt.close(fig)
        return

    summ = cph.summary[["coef","exp(coef)","exp(coef) lower 95%",
                         "exp(coef) upper 95%","p"]].copy()
    summ = summ.sort_values("coef", ascending=True)
    summ["label"]    = [LABELS.get(f, f) for f in summ.index]
    summ["modality"] = [CAUSAL_CORE.get(f, "Clinical") for f in summ.index]
    summ["color"]    = summ["modality"].map(MOD_COLOR)

    y = np.arange(len(summ))
    axes[1].scatter(summ["exp(coef)"], y, c=summ["color"], s=22, zorder=3)
    axes[1].hlines(y,
                   summ["exp(coef) lower 95%"],
                   summ["exp(coef) upper 95%"],
                   colors=summ["color"], linewidth=1.0, alpha=0.75)
    axes[1].axvline(1.0, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(summ["label"], fontsize=6.5)
    axes[1].set_xlabel("Hazard ratio (95% CI)")
    axes[1].set_title("B  Cox hazard ratios\n(penalizer=0.1, standardised continuous)")

    patches = [mpatches.Patch(color=c, label=m)
               for m, c in MOD_COLOR.items()
               if m in summ["modality"].unique()]
    axes[1].legend(handles=patches, loc="lower right",
                   fontsize=6.5, framealpha=0.8)

    plt.tight_layout(pad=0.8)
    out = OUTPUT_DIR / "fig2_feature_modalities.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# FIG 3 — KAPLAN-MEIER CURVES
# Cross-validated Cox risk score avoids in-sample leakage
# ---------------------------------------------------------------------------

def fig3_km_curves():
    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import logrank_test
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("lifelines not available — skipping fig3")
        return

    sub = df[feat_cols + ["OS.time", "OS"]].fillna(0).dropna().copy()
    T   = sub["OS.time"].values.astype(float)
    E   = sub["OS"].values.astype(int)
    X   = sub[feat_cols].values.astype(float)

    # Standardise all features before Cox
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # Cross-validated risk score — 5 folds, predict on held-out fold
    cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    risk_scores = np.zeros(len(sub))

    for train_idx, val_idx in cv.split(Xs, E):
        fold_df = pd.DataFrame(
            Xs[train_idx],
            columns=feat_cols
        )
        fold_df["OS.time"] = T[train_idx]
        fold_df["OS"]      = E[train_idx]
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(fold_df, duration_col="OS.time", event_col="OS")
            val_df = pd.DataFrame(Xs[val_idx], columns=feat_cols)
            risk_scores[val_idx] = cph.predict_log_partial_hazard(val_df).values
        except Exception:
            risk_scores[val_idx] = 0.0

    median_risk = np.median(risk_scores)
    high        = risk_scores >= median_risk
    low         = ~high

    lr = logrank_test(T[high], T[low], E[high], E[low])
    p  = lr.p_value
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 2.8))

    # Panel A: high vs low risk
    ax = axes[0]
    for mask, label, color in [
        (high, "High risk",  PALETTE["protein"]),
        (low,  "Low risk",   PALETTE["clinical"]),
    ]:
        kmf = KaplanMeierFitter()
        kmf.fit(T[mask], E[mask], label=f"{label} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12,
                                   color=color, linewidth=1.5)
    ax.set_title(
        f"A  Overall survival by risk group\n"
        f"(5-fold CV Cox risk score; log-rank {p_str})"
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)

    # Panel B: top binary feature by absolute Cox coefficient
    try:
        full_df = pd.DataFrame(Xs, columns=feat_cols)
        full_df["OS.time"] = T
        full_df["OS"]      = E
        cph_full = CoxPHFitter(penalizer=0.1)
        cph_full.fit(full_df, duration_col="OS.time", event_col="OS")

        # Force Stage IV as Panel B — clinically meaningful, validated in paper
        PREFERRED_B = [
            "CLIN_ajcc_pathologic_stage.diagnoses_Stage IV",
            "CLIN_ajcc_pathologic_m.diagnoses_M1",
            "CLIN_ajcc_pathologic_n.diagnoses_N1b",
        ]
        top_f = next((f for f in PREFERRED_B if f in feat_cols), None)
        if top_f is None:
            binary_feats = [f for f in feat_cols
                            if set(sub[f].dropna().unique()).issubset({0,1,0.0,1.0})]
            coefs = cph_full.params_[binary_feats].abs()
            top_f = coefs.idxmax()

        ax = axes[1]
        top_vals = sub[top_f].values
        for val, label, color in [
            (1, f"{LABELS.get(top_f, top_f)} = Yes", PALETTE["protein"]),
            (0, f"{LABELS.get(top_f, top_f)} = No",  PALETTE["clinical"]),
        ]:
            mask = (top_vals == val)
            if mask.sum() < 5:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(T[mask], E[mask], label=f"{label} (n={mask.sum()})")
            kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12,
                                       color=color, linewidth=1.5)
        ax.set_title(f"B  Survival by {LABELS.get(top_f, top_f)}")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.05)
    except Exception as e:
        print(f"  Panel B skipped: {e}")
        axes[1].axis("off")

    plt.tight_layout(pad=0.8)
    out = OUTPUT_DIR / "fig3_km_curves.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# FIG 4 — DIMENSIONALITY REDUCTION WATERFALL
# Values from paper text: 600k raw, ~1200 after filtering, 20 final core
# METABRIC best raw winner = 87 (external validation, separate trajectory)
# ---------------------------------------------------------------------------

def fig4_dimensionality():
    # Stage 2 input (merged candidate pool) read from mb_results_all.csv
    # Confirmed: all experiments use n_features=50 as candidate pool entering MB
    # Final core = 20 confirmed from best_strategy_features.txt (S34_cox_pen5_top20)
    MB_CSV = BASE_DIR / "01_Causal_feature_extraction" / "MB" / "mb_results_all.csv"
    stage2_input = 50   # default; overwritten if file found
    if MB_CSV.exists():
        mb_df = pd.read_csv(MB_CSV)
        stage2_input = int(mb_df["n_features"].iloc[0])
        print(f"  fig4: read stage2_input={stage2_input} from {MB_CSV.name}")
    else:
        print(f"  fig4: mb_results_all.csv not found at {MB_CSV}, using default={stage2_input}")

    # TCGA pipeline stages — confirmed numbers
    stages_tcga = [
        ("Raw multiomics\n(7 modalities)",  620_000),
        ("Merged candidate\npool (Stage 1)", stage2_input),
        ("Final causal core\n(Stage 2+Cox)",        20),
    ]

    # METABRIC pipeline — confirmed from paper Table 3
    stages_meta = [
        ("Min200\nmerged\ninput",   525),
        ("After IAMB\n(alpha=0.20)", 87),
        ("Final\npenalised\nmodel",  87),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 3.0))

    def _waterfall(ax, stages, title, ylabel="Feature count (log scale)", log=True):
        labels = [s[0] for s in stages]
        values = [s[1] for s in stages]
        x      = range(len(stages))
        colors = [PALETTE["grey"]] * (len(stages) - 1) + [PALETTE["clinical"]]
        bars   = ax.bar(x, values, color=colors, width=0.52,
                        edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, values):
            lbl = f"{v:,}" if v < 1_000_000 else f"{v//1_000:,}k+"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * (1.08 if log else 1.04),
                    lbl, ha="center", va="bottom", fontsize=7)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    # Panel A: TCGA-BRCA pipeline — stage2_input from mb_results_all.csv
    _waterfall(
        axes[0], stages_tcga,
        "A  TCGA-BRCA discovery pipeline\n"
        f"(Stage 1 count from mb_results_all.csv = {stage2_input};\n"
        "C-index=0.808, AUC$_{5yr}$=0.868)",
        log=True,
    )

    # Panel B: METABRIC — confirmed from paper Table 3
    _waterfall(
        axes[1], stages_meta,
        "B  METABRIC external validation\n"
        "(counts confirmed in paper Table 3;\n"
        "C-index=0.720, AUC$_{5yr}$=0.764)",
        ylabel="Feature count",
        log=False,
    )

    plt.tight_layout(pad=0.9)
    out = OUTPUT_DIR / "fig4_dimensionality.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# GO/KEGG GENE LISTS  (gene symbols only — no probes, no Ensembl blobs)
# Methylation probes excluded: cg IDs are not valid GO/KEGG inputs
# ---------------------------------------------------------------------------

def export_gene_lists():
    # TCGA causal core — only gene/protein symbols; probes excluded
    tcga_genes = {
        "EIF4EBP1": "4EBP1 protein — mTOR pathway effector (causal core)",
        "ZAP70":    "ZAP-70 protein — T-cell immune kinase (causal core)",
    }
    # lncRNA: submit Ensembl ID to g:Profiler (accepts ENSG IDs)
    tcga_lncrna = {
        "ENSG00000264589": "lncRNA — causal core RNA feature; submit to g:Profiler",
    }

    # METABRIC best model — annotated gene symbols (from paper Table 5)
    metabric_genes = [
        # Key oncogenic drivers (ERBB2 = HER2 gene; HER2 alias removed to avoid duplicate)
        "ERBB2", "PIK3CA",
        # Supported / emerging RNA markers
        "SPP1", "NDRG1", "PRAME", "SERPINE1", "S100P", "LAD1",
        # Context-dependent mutation markers
        "RUNX1", "BRCA2",
        # Additional mutation markers (exploratory)
        "ROS1", "UBR5", "ARID2", "FANCA", "TAF1",
        "ARID1A", "BRIP1", "MEN1",
    ]

    # TCGA gene-symbol file
    out1 = OUTPUT_DIR / "go_genes_tcga_core.txt"
    with open(out1, "w") as f:
        f.write("# GO/KEGG gene list — TCGA-BRCA causal core (molecular features)\n")
        f.write("# Submit to: https://biit.cs.ut.ee/gprofiler/gost\n")
        f.write("# Note: methylation probes (cg*) excluded — not valid gene identifiers\n\n")
        f.write("# Gene symbols (for most GO/KEGG tools)\n")
        for g, note in tcga_genes.items():
            f.write(f"{g}\t# {note}\n")
        f.write("\n# Ensembl ID (for g:Profiler or Ensembl-aware tools)\n")
        for g, note in tcga_lncrna.items():
            f.write(f"{g}\t# {note}\n")
    print(f"Saved: {out1.name}  ({len(tcga_genes)+len(tcga_lncrna)} entries)")

    # METABRIC gene-symbol file
    out2 = OUTPUT_DIR / "go_genes_metabric.txt"
    with open(out2, "w") as f:
        f.write("# GO/KEGG gene list — METABRIC external validation model\n")
        f.write("# Source: paper Table 5, annotated gene symbols only\n")
        f.write("# Submit to: https://bioinformatics.sdstate.edu/go/\n\n")
        for g in metabric_genes:
            f.write(f"{g}\n")
    print(f"Saved: {out2.name}  ({len(metabric_genes)} entries)")

    # Excluded features explanation
    out3 = OUTPUT_DIR / "non_gene_features_excluded.txt"
    with open(out3, "w") as f:
        f.write("# Features excluded from GO/KEGG submission\n")
        f.write("# These are not valid gene identifiers for enrichment tools\n\n")
        for feat, mod in CAUSAL_CORE.items():
            if mod == "Clinical":
                f.write(f"EXCLUDED  Clinical variable: {LABELS.get(feat, feat)}\n")
            elif feat.startswith("METH_"):
                probe = feat.replace("METH_", "")
                f.write(f"EXCLUDED  Methylation probe: {probe} "
                        f"(map to gene via UCSC/Illumina manifest first)\n")
    print(f"Saved: {out3.name}")


# ---------------------------------------------------------------------------
# LATEX TABLE 1 — cohort summary from actual data
# ---------------------------------------------------------------------------

def export_latex_table():
    n          = len(df)
    n_ev       = int(df["OS"].sum())
    ev_rate    = f"{n_ev/n*100:.1f}\\%"
    med_time   = f"{df['OS.time'].median():.0f}"
    n_feat     = len(feat_cols)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary of TCGA-BRCA cohort and causal core}",
        r"\label{tab:cohort}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Characteristic & Value \\",
        r"\midrule",
        f"Patients & {n} \\\\",
        f"Overall survival events & {n_ev} ({ev_rate}) \\\\",
        f"Median follow-up (days) & {med_time} \\\\",
        f"Causal core features & {n_feat} \\\\",
        r"Modalities represented & Clinical, Protein, Methylation, RNA \\",
        r"\midrule",
        r"Model (S34) C-index & 0.808 \\",
        r"Model (S34) 5-year AUC & 0.868 \\",
        r"\midrule",
        r"METABRIC validation (n) & 1,980 \\",
        r"METABRIC event rate & 57.7\% \\",
        r"METABRIC C-index & 0.720 \\",
        r"METABRIC 5-year AUC & 0.764 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUTPUT_DIR / "table_cohort.tex"
    out.write_text("\n".join(lines))
    print(f"Saved: {out.name}")



# ---------------------------------------------------------------------------
# FIG 5 — SHAP BEESWARM (surrogate classifier on 5-year mortality)
# Model: GradientBoosting trained on Y_died_5yr (binary 5-year endpoint)
# using the 20-feature causal core.
# This is a surrogate interpretation model, not the primary Cox model.
# Labeled accordingly per journal transparency requirements.
# ---------------------------------------------------------------------------

def fig5_shap_beeswarm():
    try:
        import shap
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("shap not available — skipping fig5. Run: pip install shap")
        return

    # Use causal_features_dataset (has all 20 causal core features) + derive 5-year label
    # ite_ready_dataset excludes the two treatment columns used as T → only 18 features
    # causal_features_dataset has all 23 columns including both treatment columns
    ite_path = BASE_DIR / "02_ITE" / "01_preprocessing" / "output" / "ite_ready_dataset.csv"
    if ite_path.exists():
        ite_df = pd.read_csv(ite_path, na_values=TCGA_NA, low_memory=False)
        y_col  = "Y"
        src    = "ite_ready_dataset (Y = died within 5 years)"
    else:
        ite_df = None

    # For features: always use causal_features_dataset which has all 20
    # Merge outcome from ite_ready into causal_features_dataset
    feat_df = df.copy()   # df = causal_features_dataset loaded at module level
    if ite_df is not None and "Y" in ite_df.columns and "patient_id" in ite_df.columns:
        y_map = ite_df.set_index("patient_id")["Y"]
        if "patient_id" in feat_df.columns:
            feat_df["Y_5yr"] = feat_df["patient_id"].map(y_map).fillna(0).astype(int)
        else:
            feat_df["Y_5yr"] = ((feat_df["OS"] == 1) &
                                (feat_df["OS.time"] <= 5*365)).astype(int)
    else:
        feat_df["Y_5yr"] = ((feat_df["OS"] == 1) &
                            (feat_df["OS.time"] <= 5*365)).astype(int)
    y_col = "Y_5yr"
    src   = "causal_features_dataset + 5-year label from ite_ready"

    feat_available = [f for f in CAUSAL_CORE if f in feat_df.columns]
    if len(feat_available) < 10:
        print(f"Only {len(feat_available)} features found — skipping SHAP")
        return

    sub   = feat_df[feat_available + [y_col]].fillna(0).dropna()
    X     = sub[feat_available].values.astype(float)
    y     = sub[y_col].astype(int).values

    print(f"  SHAP source     : {src}")
    print(f"  n={len(sub)}, Y=1: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Features used   : {len(feat_available)}")

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    # GBC returns array of shape (n, features) for binary case
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1]   # class=1 (death)
    else:
        shap_arr = shap_vals

    # Order by mean |SHAP| descending
    mean_abs  = np.abs(shap_arr).mean(axis=0)
    order     = np.argsort(mean_abs)   # ascending → bottom to top in beeswarm

    ordered_feats  = [feat_available[i] for i in order]
    ordered_labels = [LABELS.get(f, f) for f in ordered_feats]
    ordered_colors = [MOD_COLOR[CAUSAL_CORE.get(f, "Clinical")] for f in ordered_feats]

    fig, ax = plt.subplots(figsize=(FIG_DOUBLE, max(4.0, len(feat_available)*0.28)))

    shap.summary_plot(
        shap_arr[:, order],
        X[:, order],
        feature_names=ordered_labels,
        plot_type="dot",
        max_display=len(feat_available),
        show=False,
        color_bar=True,
        plot_size=None,
        alpha=0.6,
    )

    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on predicted 5-year mortality risk)", fontsize=8)
    ax.set_title(
        f"Supplementary: SHAP feature contributions — 20-feature causal core\n"
        f"(Surrogate GradientBoosting classifier on 5-year mortality label;\n"
        f"not the primary Cox survival endpoint of Paper 1; n={len(sub)})",
        fontsize=7.5
    )

    # Color y-tick labels by modality
    label_to_feat = {LABELS.get(f, f): f for f in feat_available}
    for tl in ax.get_yticklabels():
        feat = label_to_feat.get(tl.get_text())
        if feat:
            tl.set_color(MOD_COLOR[CAUSAL_CORE.get(feat, "Clinical")])

    # Modality legend
    present_mods = {CAUSAL_CORE.get(f, "Clinical") for f in feat_available}
    patches = [mpatches.Patch(color=MOD_COLOR[m], label=m)
               for m in ["Clinical","Protein","Methylation","RNA"]
               if m in present_mods]
    ax.legend(handles=patches, loc="lower right", fontsize=6.5,
              framealpha=0.8, title="Modality")

    plt.tight_layout(pad=0.6)
    out = OUTPUT_DIR / "figS1_shap_beeswarm_supplementary.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# FIG 6 — SHAP on 50-feature MB candidate pool (pre-Cox)
# Reads best_config.json to get dataset name, then loads the merged CSV
# to get the 50-feature intermediate set selected by Stage 2 MB.
# Complements figS1 (20 features post-Cox) by showing what MB selected
# before Cox penalization removed 30 less-prognostic features.
# ---------------------------------------------------------------------------

def fig6_shap_mb50():
    try:
        import shap
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        print("shap not available — skipping fig6. Run: pip install shap")
        return

    # Read best_config.json to find the best MB dataset
    cfg_path = (BASE_DIR / "01_Causal_feature_extraction" / "MB"
                / "best_config.json")
    if not cfg_path.exists():
        print(f"  fig6: best_config.json not found at {cfg_path} — skipping")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)

    print(f"  fig6: best_config = {cfg}")

    # Find the merged dataset file
    # Try multiple candidate locations based on Thesis_v3 folder structure
    merge_base = BASE_DIR / "01_Causal_feature_extraction"
    dataset_key = str(cfg.get("best_experiment", cfg.get("dataset", "08_composite")))

    candidate_paths = [
        merge_base / "MERGE_outer"            / f"{dataset_key}.csv",
        merge_base / "MERGE_continuous_outer" / f"{dataset_key}.csv",
        merge_base / "MERGE_outer"            / f"{dataset_key}_merged.csv",
        merge_base / "MB" / "results_fallback_experiment" / "best_dataset.csv",
    ]

    merged_path = next((p for p in candidate_paths if p.exists()), None)
    if merged_path is None:
        print(f"  fig6: cannot find merged dataset for key='{dataset_key}'")
        print(f"  Tried: {[str(p) for p in candidate_paths]}")
        print("  Provide the correct merged CSV path and re-run.")
        return

    merged_df = pd.read_csv(merged_path, na_values=TCGA_NA, low_memory=False)
    print(f"  fig6: loaded {merged_path.name} — shape={merged_df.shape}")

    # Use the exact 50 features from best_config.json (already logged there)
    feat50_from_cfg = cfg.get("features", [])
    if feat50_from_cfg:
        feat50 = [f for f in feat50_from_cfg if f in merged_df.columns]
        print(f"  fig6: using {len(feat50)} / {len(feat50_from_cfg)} features from best_config.json")
    else:
        # Fallback: exclude known non-feature columns
        exclude = {"Y","OS","OS.time","patient_id","_PATIENT","propensity_score",
                   "T","T_hormone","T_chemo","T_targeted","T_radiation","T_hormone_excl",
                   "Unnamed: 0"}
        feat50 = [c for c in merged_df.columns
                  if c not in exclude
                  and pd.api.types.is_numeric_dtype(merged_df[c])]
        print(f"  fig6: using {len(feat50)} numeric features (fallback)")

    # Get outcome: prefer 5-year binary label
    y_col = next((c for c in ["Y", "Y_died_5yr"] if c in merged_df.columns), None)
    if y_col is None and "OS" in merged_df.columns:
        if "OS.time" in merged_df.columns:
            merged_df["Y_5yr"] = ((merged_df["OS"] == 1) &
                                  (merged_df["OS.time"] <= 5*365)).astype(int)
        else:
            merged_df["Y_5yr"] = merged_df["OS"].astype(int)
        y_col = "Y_5yr"
    if y_col is None:
        print("  fig6: no outcome column found — skipping")
        return
    print(f"  fig6: outcome={y_col}, Y=1: {int(merged_df[y_col].fillna(0).sum())} "
          f"({merged_df[y_col].fillna(0).mean()*100:.1f}%)")

    sub  = merged_df[feat50 + [y_col]].fillna(0).dropna()
    X50  = sub[feat50].values.astype(float)
    y50  = sub[y_col].astype(int).values

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )
    clf.fit(X50, y50)

    explainer = shap.TreeExplainer(clf)
    sv        = explainer.shap_values(X50)
    shap_arr  = sv[1] if isinstance(sv, list) else sv

    # Order by mean |SHAP|
    mean_abs = np.abs(shap_arr).mean(axis=0)
    order    = np.argsort(mean_abs)
    top_n    = min(30, len(feat50))   # show top 30 for readability
    order    = order[-top_n:]         # top-n ascending (beeswarm shows bottom→top)

    # Color by modality prefix
    def modality_color(feat):
        if feat.startswith("CLIN_"):   return PALETTE["clinical"]
        if feat.startswith("PROT_"):   return PALETTE["protein"]
        if feat.startswith("METH_"):   return PALETTE["methylation"]
        if feat.startswith("RNA_"):    return PALETTE["rna"]
        if feat.startswith("MUT_"):    return "#9467BD"
        if feat.startswith("CNV_"):    return "#8C564B"
        if feat.startswith("MIRNA_"):  return "#17BECF"
        return PALETTE["grey"]

    ordered_feats  = [feat50[i] for i in order]
    ordered_labels = [f[:45] for f in ordered_feats]   # truncate long names
    dot_colors     = [modality_color(f) for f in ordered_feats]

    fig, ax = plt.subplots(figsize=(FIG_DOUBLE, max(5.0, top_n * 0.27)))

    shap.summary_plot(
        shap_arr[:, order],
        X50[:, order],
        feature_names=ordered_labels,
        plot_type="dot",
        max_display=top_n,
        show=False,
        color_bar=True,
        plot_size=None,
        alpha=0.55,
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on predicted outcome)", fontsize=8)
    ax.set_title(
        f"Supplementary: SHAP — MB Stage 2 candidate pool (top {top_n} of {len(feat50)} features)\n"
        f"(dataset: {dataset_key}; surrogate GradientBoosting classifier; n={len(sub)})\n"
        "Complements figS1 (final 20-feature causal core after Cox penalization)",
        fontsize=7.5
    )

    # Modality legend
    mod_legend = {
        "Clinical":    PALETTE["clinical"],
        "Protein":     PALETTE["protein"],
        "Methylation": PALETTE["methylation"],
        "RNA":         PALETTE["rna"],
        "Mutation":    "#9467BD",
        "CNV":         "#8C564B",
        "miRNA":       "#17BECF",
    }
    patches = [mpatches.Patch(color=c, label=m) for m, c in mod_legend.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=6, framealpha=0.8,
              title="Modality prefix")

    plt.tight_layout(pad=0.6)
    out = OUTPUT_DIR / "figS2_shap_mb50_supplementary.png"
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nOutput: {OUTPUT_DIR}\n")

    fig1_cohort_outcomes()
    fig2_feature_modalities()
    fig3_km_curves()
    fig4_dimensionality()
    fig5_shap_beeswarm()    # Supplementary S1: 20-feature causal core
    fig6_shap_mb50()        # Supplementary S2: 50-feature MB candidate pool
    export_gene_lists()
    export_latex_table()

    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print("\nFiles:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")