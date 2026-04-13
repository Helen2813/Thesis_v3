"""
Final ITE Comparison: LinearDML vs CausalForestDML
===================================================
Thesis: Causal Multimodal Analysis of Breast Cancer Survival (TCGA-BRCA)

Script:  Thesis_v3/02_ITE/05_final_comparison/final_ite_comparison.py
Run:     python final_ite_comparison.py
Install: pip install econml tqdm lifelines
"""

import os, sys, json, warnings, logging
from pathlib import Path
from time import time
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

try:
    from econml.dml import LinearDML, CausalForestDML
    from econml.score import RScorer
except ImportError:
    log.error("Run: pip install econml"); sys.exit(1)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})
RANDOM_STATE = 42
N_BOOT       = 2000
CV_FOLDS     = 5
COLORS       = {'LinearDML': '#4C72B0', 'CausalForestDML': '#DD8452'}

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 1/8  Loading data")
print("="*65)

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR   = SCRIPT_DIR.parent.parent
INPUT_DIR  = BASE_DIR / '02_ITE' / '01_preprocessing' / 'output'
OUTPUT_DIR = SCRIPT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ite_df = pd.read_csv(INPUT_DIR / 'ite_ready_dataset.csv')
with open(INPUT_DIR / 'preprocessing_metadata.json') as f:
    meta = json.load(f)

CLASS_WEIGHT = {int(k): float(v)
                for k,v in meta.get('class_weight_balanced',
                                    {'0':0.55,'1':5.38}).items()}

X_df  = ite_df.drop(columns=['T','Y','patient_id','propensity_score'],
                    errors='ignore')
FEAT  = X_df.columns.tolist()
X     = X_df.values.astype(float)
Y     = ite_df['Y'].astype(int).values
T_bin = ite_df['T'].astype(int).values

print(f"  Patients   : {X.shape[0]}")
print(f"  Features   : {X.shape[1]}")
print(f"  Y=1 events : {Y.sum()} ({Y.mean()*100:.1f}%)")
print(f"  T=1 treated: {T_bin.sum()} ({T_bin.mean()*100:.1f}%)")
print(f"  Class weight: {CLASS_WEIGHT}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MULTI-TREATMENT MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 2/8  Building multi-treatment matrix")
print("="*65)

TREATMENT_LABELS = {
    'T_any':      'Any Treatment',
    'T_hormone':  'Hormone Therapy (ER+/PR+)',
    'T_targeted': 'Targeted Therapy (HER2+)',
    'T_chemo':    'Chemotherapy (TNBC)',
}

T_matrix = pd.DataFrame({'T_any': T_bin})
has_receptor = all(c in X_df.columns for c in ['ER_status','PR_status','HER2_status'])

if has_receptor:
    er   = (X_df['ER_status'].fillna(0).values   > 0.5).astype(int)
    pr   = (X_df['PR_status'].fillna(0).values   > 0.5).astype(int)
    her2 = (X_df['HER2_status'].fillna(0).values > 0.5).astype(int)

    # Subgroup masks: each arm is evaluated ONLY within its relevant patient subgroup.
    # Control = untreated patients from the SAME molecular subtype — not entire population.
    # This avoids the confounding where e.g. TNBC patients (worse prognosis, untreated)
    # become the control group for hormone therapy, inflating apparent harm.
    mask_hormone  = ((er | pr) & (her2 == 0)).astype(bool)        # ER+/PR+ HER2-
    mask_targeted = (her2 == 1).astype(bool)                       # HER2+
    mask_chemo    = ((er==0) & (pr==0) & (her2==0)).astype(bool)   # TNBC
    mask_any      = np.ones(len(T_bin), dtype=bool)                # all patients

    T_matrix['T_hormone']  = T_bin.copy()
    T_matrix['T_targeted'] = T_bin.copy()
    T_matrix['T_chemo']    = T_bin.copy()

    SUBGROUP_MASKS = {
        'T_any':      mask_any,
        'T_hormone':  mask_hormone,
        'T_targeted': mask_targeted,
        'T_chemo':    mask_chemo,
    }
    T_cols = ['T_any', 'T_hormone', 'T_targeted', 'T_chemo']
else:
    mask_any = np.ones(len(T_bin), dtype=bool)
    SUBGROUP_MASKS = {'T_any': mask_any}
    T_cols = ['T_any']
    log.warning("Receptor columns absent — using T_any only")

for tc in tqdm(T_cols, desc="  Validating arms"):
    n1 = T_matrix[tc].sum()
    n0 = int((T_matrix[tc][SUBGROUP_MASKS[tc]]==0).sum())
    n1 = int(T_matrix[tc][SUBGROUP_MASKS[tc]].sum())
    print(f"  {tc:<14} treated={n1:>4}  control={n0:>4}  "
          f"→ {TREATMENT_LABELS[tc]}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BASE ESTIMATORS
# ═══════════════════════════════════════════════════════════════════════════════
def rf_clf():
    # class_weight='balanced' required when T is multi-output (multiple treatment arms)
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=RANDOM_STATE)

def rf_reg():
    return RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3+4+5 — FIT PER SUBGROUP + EXTRACT ITE
# Each arm fitted only on patients from that molecular subtype.
# Control = untreated patients of SAME subtype — not entire population.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 3-5/8  Fitting models per subgroup & extracting ITE")
print("="*65)

all_ite   = {"LinearDML": {}, "CausalForestDML": {}}
all_ci    = {"LinearDML": {}, "CausalForestDML": {}}
fitted_models = {}

total_arms = len(T_cols) * 2
with tqdm(total=total_arms, desc="  Fitting arms") as pbar:
    for tc in T_cols:
        mask = SUBGROUP_MASKS[tc]
        Xs   = X[mask]
        Ts   = T_matrix[tc].values[mask].astype(float)
        Ys   = Y[mask].astype(float)
        n1   = int(Ts.sum())
        n0   = int((Ts == 0).sum())

        if n1 < 15 or n0 < 15:
            log.warning(f"Skipping {tc}: n1={n1} n0={n0}")
            pbar.update(2); continue

        # LinearDML
        lin = LinearDML(model_y=rf_reg(), model_t=rf_reg(),
                        linear_first_stages=False,
                        cv=CV_FOLDS, random_state=RANDOM_STATE)
        lin.fit(Ys, Ts, X=Xs, inference="statsmodels")
        ite_lin_sub  = lin.effect(Xs, T0=np.zeros(len(Xs)), T1=np.ones(len(Xs))).flatten()
        ite_lin_full = np.full(len(X), np.nan); ite_lin_full[mask] = ite_lin_sub
        all_ite["LinearDML"][tc] = ite_lin_full
        fitted_models[f"LinearDML_{tc}"] = lin
        log.info(f"  LinearDML|{tc} n={mask.sum()}  ATE={ite_lin_sub.mean():+.5f}  "
                 f"benefit={(ite_lin_sub<0).mean()*100:.1f}%")
        pbar.update(1)

        # CausalForestDML
        cf = CausalForestDML(model_y=rf_reg(), model_t=rf_reg(),
                             n_estimators=300, min_samples_leaf=10,
                             max_features="sqrt", cv=CV_FOLDS,
                             random_state=RANDOM_STATE, n_jobs=-1)
        cf.fit(Ys, Ts, X=Xs)
        ite_cf_sub  = cf.effect(Xs, T0=np.zeros(len(Xs)), T1=np.ones(len(Xs))).flatten()
        ite_cf_full = np.full(len(X), np.nan); ite_cf_full[mask] = ite_cf_sub
        all_ite["CausalForestDML"][tc] = ite_cf_full
        fitted_models[f"CausalForestDML_{tc}"] = cf
        inf    = cf.effect_inference(Xs, T0=np.zeros(len(Xs)), T1=np.ones(len(Xs)))
        lo_s   = inf.conf_int(alpha=0.05)[0].flatten()
        hi_s   = inf.conf_int(alpha=0.05)[1].flatten()
        lo_f   = np.full(len(X), np.nan); lo_f[mask] = lo_s
        hi_f   = np.full(len(X), np.nan); hi_f[mask] = hi_s
        all_ci["CausalForestDML"][tc] = (lo_f, hi_f)
        log.info(f"  CausalForestDML|{tc} n={mask.sum()}  ATE={ite_cf_sub.mean():+.5f}  "
                 f"benefit={(ite_cf_sub<0).mean()*100:.1f}%")
        pbar.update(1)
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 6/8  Computing metrics")
print("="*65)

def bootstrap_ci_fn(arr, mask=None, n=N_BOOT, seed=RANDOM_STATE):
    a   = arr[mask] if mask is not None else arr
    rng = np.random.default_rng(seed)
    b   = [rng.choice(a, len(a), replace=True).mean() for _ in range(n)]
    return float(a.mean()), float(np.percentile(b,2.5)), float(np.percentile(b,97.5))

def compute_auuc_qini(ite, T, Y, n_bins=100):
    order = np.argsort(-ite)
    T_o, Y_o = T[order], Y[order]
    n = len(T_o)
    fracs, uplifts = [], []
    for ki in range(1, n_bins+1):
        tk = int(n * ki / n_bins)
        sT, sY = T_o[:tk], Y_o[:tk]
        n1, n0 = int(sT.sum()), int((sT==0).sum())
        if n1==0 or n0==0:
            fracs.append(ki/n_bins); uplifts.append(np.nan); continue
        fracs.append(ki/n_bins)
        uplifts.append(sY[sT==1].mean() - sY[sT==0].mean())
    fracs   = np.array(fracs)
    uplifts = np.array(uplifts)
    valid   = ~np.isnan(uplifts)
    auuc    = float(np.trapz(uplifts[valid], fracs[valid]))
    raw_ate = Y[T==1].mean() - Y[T==0].mean()
    qini    = auuc - raw_ate * 0.5
    return fracs, uplifts, auuc, qini

def policy_value(ite, T, Y, X):
    lr  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X, T)
    ps  = lr.predict_proba(X)[:,1].clip(0.05, 0.95)
    ipw1 = np.where(T==1, Y/ps, 0)
    ipw0 = np.where(T==0, Y/(1-ps), 0)
    pol  = (ite < 0).astype(int)
    return {
        'policy':     float((pol*ipw1 + (1-pol)*ipw0).mean()),
        'treat_all':  float(ipw1.mean()),
        'treat_none': float(ipw0.mean()),
    }

def r_score_fn(ite, X, T, Y):
    try:
        scorer = RScorer(model_y=rf_reg(), model_t=rf_reg(),  # reg for continuous T residuals
                         cv=CV_FOLDS, random_state=RANDOM_STATE)
        scorer.fit(Y, T, X=X)
        return float(scorer.score(ite.reshape(-1,1), X))
    except Exception as ex:
        log.warning(f"R-score skipped: {ex}"); return np.nan

metrics_all = {}
total_jobs  = len(all_ite) * len(T_cols)

with tqdm(total=total_jobs, desc="  Computing metrics") as pbar:
    for mname in all_ite:
        metrics_all[mname] = {}
        for tc in T_cols:
            mask  = SUBGROUP_MASKS[tc]
            ite_full = all_ite[mname][tc]
            ite   = ite_full[mask]
            T_arm = T_matrix[tc].values[mask].astype(int)
            Y_arm = Y[mask]
            X_arm = X[mask]

            ate, ate_lo, ate_hi = bootstrap_ci_fn(ite)
            att, att_lo, att_hi = bootstrap_ci_fn(ite, mask=T_arm==1)
            atc, atc_lo, atc_hi = bootstrap_ci_fn(ite, mask=T_arm==0)
            fracs, uplifts, auuc, qini = compute_auuc_qini(ite, T_arm, Y_arm)
            pv = policy_value(ite, T_arm, Y_arm, X_arm)

            m = {
                'ite': ite, 'fracs': fracs, 'uplifts': uplifts,
                # ── Effect estimates ──
                'ATE': ate, 'ATE_lo': ate_lo, 'ATE_hi': ate_hi,
                'ATT': att, 'ATT_lo': att_lo, 'ATT_hi': att_hi,
                'ATC': atc, 'ATC_lo': atc_lo, 'ATC_hi': atc_hi,
                # ── Ranking / uplift ──
                'AUUC': auuc, 'Qini': qini,
                # ── Policy ──
                'policy':    pv['policy'],
                'treat_all': pv['treat_all'],
                'treat_none':pv['treat_none'],
                # ── Heterogeneity ──
                'pct_benefit': float((ite<0).mean()*100),
                'pct_harm':    float((ite>0).mean()*100),
                'ITE_std':     float(ite.std()),
                'ITE_IQR':     float(np.percentile(ite,75)-np.percentile(ite,25)),
            }
            if mname == 'CausalForestDML' and tc in all_ci[mname]:
                lo_full, hi_full = all_ci[mname][tc]
                lo = lo_full[mask]; hi = hi_full[mask]
                m['ite_lo'] = lo; m['ite_hi'] = hi
                m['pct_sig_ITE'] = float(((lo>0)|(hi<0)).mean()*100)

            metrics_all[mname][tc] = m
            pbar.update(1)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 7/8  Results tables")
print("="*65)

rows = []
for mname in all_ite:
    for tc in T_cols:
        m   = metrics_all[mname][tc]
        # FIX: recompute T_arm per arm — previously used stale value from last loop iteration
        _mask  = SUBGROUP_MASKS[tc]
        _T_arm = T_matrix[tc].values[_mask].astype(int)
        sig_ate = '✓' if not (m['ATE_lo']<0<m['ATE_hi']) else '○'
        sig_att = '✓' if not (m['ATT_lo']<0<m['ATT_hi']) else '○'
        row = {
            'Model':            mname,
            'Treatment Arm':    TREATMENT_LABELS[tc],
            'N treated':        int(_T_arm.sum()),
            'N control':        int((_T_arm==0).sum()),
            # Effect estimates
            'ATE':              round(m['ATE'],5),
            'ATE_95CI':         f"[{m['ATE_lo']:+.4f}, {m['ATE_hi']:+.4f}]",
            'ATT':              round(m['ATT'],5),
            'ATT_95CI':         f"[{m['ATT_lo']:+.4f}, {m['ATT_hi']:+.4f}]",
            'ATC':              round(m['ATC'],5),
            # Ranking quality
            'AUUC':             round(m['AUUC'],6),
            'Qini':             round(m['Qini'],6),
            # Policy
            'Policy_value':     round(m['policy'],4),
            'Treat_all':        round(m['treat_all'],4),
            'Treat_none':       round(m['treat_none'],4),
            'Policy_gain':      round(m['treat_none']-m['policy'],5),
            # Heterogeneity
            'Benefit_%':        round(m['pct_benefit'],1),
            'Harm_%':           round(m['pct_harm'],1),
            'ITE_std':          round(m['ITE_std'],5),
            'ITE_IQR':          round(m['ITE_IQR'],5),
            # CausalForest only
            'Sig_ITE_%':        round(m.get('pct_sig_ITE', np.nan),1),
            'ATE_sig':          sig_ate,
            'ATT_sig':          sig_att,
        }
        rows.append(row)

results_df = pd.DataFrame(rows)
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / 'final_comparison_table.csv', index=False)

# Separate wide table for thesis (one row per arm, models as columns)
thesis_rows = []
for tc in T_cols:
    r = {'Treatment Arm': TREATMENT_LABELS[tc]}
    for mname in all_ite:
        m = metrics_all[mname][tc]
        pfx = 'LIN' if mname == 'LinearDML' else 'CF'
        r[f'{pfx}_ATE']       = round(m['ATE'],5)
        r[f'{pfx}_ATE_CI']    = f"[{m['ATE_lo']:+.4f},{m['ATE_hi']:+.4f}]"
        r[f'{pfx}_AUUC']      = round(m['AUUC'],5)
        r[f'{pfx}_Qini']      = round(m['Qini'],5)
        r[f'{pfx}_Benefit_%'] = round(m['pct_benefit'],1)
    thesis_rows.append(r)

thesis_df = pd.DataFrame(thesis_rows)
thesis_df.to_csv(OUTPUT_DIR / 'thesis_table_wide.csv', index=False)
print("\n=== Thesis Table (wide format) ===")
print(thesis_df.to_string(index=False))

# LinearDML coefficients
print("\n=== LinearDML: Top Treatment Effect Modifiers ===")
try:
    lin_model_any = fitted_models.get('LinearDML_T_any')
    if lin_model_any is None:
        raise ValueError('LinearDML_T_any not in fitted_models')
    coefs = lin_model_any.coef_ if not callable(lin_model_any.coef_) else lin_model_any.coef_()
    coefs = np.array(coefs)
    if coefs.ndim == 1: coefs = coefs.reshape(1,-1)
    coef_rows = []
    for j, tc in enumerate(T_cols):
        # Use per-arm LinearDML model for arm-specific coefficients
        lin_arm = fitted_models.get(f'LinearDML_{tc}', lin_model_any)
        try:
            c_arm = lin_arm.coef_ if not callable(lin_arm.coef_) else lin_arm.coef_()
            row_c = np.array(c_arm).flatten()
        except Exception:
            row_c = np.array(coefs[j] if j < len(coefs) else coefs[0]).flatten()
        for feat, c in zip(FEAT, row_c):
            coef_rows.append({'arm': TREATMENT_LABELS[tc],
                               'feature': feat,
                               'coef': float(c),
                               'abs_coef': float(abs(c))})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUTPUT_DIR / 'lineardml_coefs.csv', index=False)
    for tc in T_cols:
        sub = (coef_df[coef_df['arm']==TREATMENT_LABELS[tc]]
               .sort_values('abs_coef', ascending=False).head(5))
        print(f"\n  {TREATMENT_LABELS[tc]}:")
        print(sub[['feature','coef']].to_string(index=False))
except Exception as ex:
    log.warning(f"LinearDML coefs: {ex}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STEP 8/8  Generating figures")
print("="*65)

def save_fig(fig, name):
    p = OUTPUT_DIR / name
    fig.savefig(p, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {p.name}")

n_arms  = len(T_cols)
m_names = list(all_ite.keys())

fig_jobs = ['ATE comparison', 'Uplift curves', 'ITE distributions',
            'LinearDML coefs', 'CausalForest CI', 'Policy value']

with tqdm(total=len(fig_jobs), desc="  Figures") as pfig:

    # ── Fig 1: ATE comparison ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_arms, figsize=(5*n_arms+1, 5), squeeze=False)
    axes = axes[0]
    for ax, tc in zip(axes, T_cols):
        names = m_names
        ates  = [metrics_all[n][tc]['ATE']    for n in names]
        lo_e  = [metrics_all[n][tc]['ATE'] - metrics_all[n][tc]['ATE_lo'] for n in names]
        hi_e  = [metrics_all[n][tc]['ATE_hi'] - metrics_all[n][tc]['ATE'] for n in names]
        bars  = ax.bar(names, ates,
                       color=[COLORS[n] for n in names],
                       alpha=0.85, edgecolor='white', linewidth=1.2)
        ax.errorbar(names, ates, yerr=[lo_e, hi_e],
                    fmt='none', color='black', capsize=6, linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.2)
        _m2  = SUBGROUP_MASKS[tc]
        _T2  = T_matrix[tc].values[_m2].astype(int)
        _n1  = int(_T2.sum()); _n0 = int((_T2==0).sum())
        ax.set_title(f"{TREATMENT_LABELS[tc][:28]}\n(treated={_n1}, control={_n0})",
                     fontsize=9, fontweight='bold')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('ATE (\u0394 mortality risk)' if ax is axes[0] else '')
        for bar, val in zip(bars, ates):
            ax.text(bar.get_x()+bar.get_width()/2,
                    val + 0.001 if val>=0 else val-0.003,
                    f'{val:+.4f}', ha='center', fontsize=8)
    plt.suptitle('ATE by Treatment Arm: LinearDML vs CausalForestDML\n'
                 '(95% CI, multi-treatment, 5-fold cross-fitted)', fontsize=11)
    plt.tight_layout()
    save_fig(fig, 'fig1_ate_comparison.png')
    pfig.update(1)

    # ── Fig 2: Uplift curves ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_arms, figsize=(5*n_arms+1, 5), squeeze=False)
    axes = axes[0]
    for ax, tc in zip(axes, T_cols):
        T_arm = T_matrix[tc].values.astype(int)
        raw   = Y[T_arm==1].mean() - Y[T_arm==0].mean()
        for mname in list(all_ite.keys()):
            m     = metrics_all[mname][tc]
            valid = ~np.isnan(m['uplifts'])
            ax.plot(m['fracs'][valid], m['uplifts'][valid],
                    color=COLORS[mname], linewidth=2,
                    label=f"{mname}  AUUC={m['AUUC']:.4f}  Qini={m['Qini']:.4f}")
        ax.axhline(raw, color='gray', linestyle='--', linewidth=1.2,
                   label=f'Random (raw={raw:+.4f})')
        ax.set_xlabel('Fraction targeted (ITE rank)')
        ax.set_ylabel('Δ outcome' if ax is axes[0] else '')
        ax.set_title(TREATMENT_LABELS[tc][:30], fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
    plt.suptitle('Uplift Curves — LinearDML vs CausalForestDML', fontsize=11)
    plt.tight_layout()
    save_fig(fig, 'fig2_uplift_curves.png')
    pfig.update(1)

    # ── Fig 3: ITE distributions ─────────────────────────────────────────────
    fig, axes = plt.subplots(len(m_names), n_arms,
                              figsize=(5*n_arms, 4*len(m_names)),
                              squeeze=False)
    for row, mname in enumerate(m_names):
        for col, tc in enumerate(T_cols):
            ax  = axes[row][col]
            ite = metrics_all[mname][tc]['ite']
            ax.hist(ite, bins=35, color=COLORS[mname],
                    alpha=0.75, edgecolor='white', linewidth=0.5)
            ax.axvline(0,          color='gray',  linestyle='--', lw=1.5)
            ax.axvline(ite.mean(), color='black', linestyle='-',  lw=2,
                       label=f'ATE={ite.mean():+.4f}')
            # CausalForest: add mean CI bounds
            if mname == 'CausalForestDML' and tc in all_ci[mname]:
                lo, hi = all_ci[mname][tc]
                ax.axvline(lo.mean(), color='steelblue',
                           linestyle=':', lw=1.5, label=f'CI lo={lo.mean():+.4f}')
                ax.axvline(hi.mean(), color='steelblue',
                           linestyle=':', lw=1.5, label=f'CI hi={hi.mean():+.4f}')
            pct = metrics_all[mname][tc]['pct_benefit']
            ax.text(0.03, 0.96, f'Benefit: {pct:.1f}%',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_title(f'{mname}\n{TREATMENT_LABELS[tc][:25]}',
                         fontsize=8, fontweight='bold')
            ax.set_xlabel('ITE'); ax.set_ylabel('Patients')
            ax.legend(fontsize=7)
    plt.suptitle('ITE Distributions — All Arms × Both Models', fontsize=11)
    plt.tight_layout()
    save_fig(fig, 'fig3_ite_distributions.png')
    pfig.update(1)

    # ── Fig 4: LinearDML coefficients ────────────────────────────────────────
    try:
        lin_model_any = fitted_models.get('LinearDML_T_any')
        if lin_model_any is None:
            raise ValueError('LinearDML_T_any not fitted')
        coefs = lin_model_any.coef_ if not callable(lin_model_any.coef_) else lin_model_any.coef_()
        coefs = np.array(coefs)
        if coefs.ndim == 1: coefs = coefs.reshape(1,-1)
        fig, axes = plt.subplots(1, n_arms, figsize=(5*n_arms+1, 5), squeeze=False)
        axes = axes[0]
        for ax, (j, tc) in zip(axes, enumerate(T_cols)):
            lin_arm = fitted_models.get(f'LinearDML_{tc}', lin_model_any)
            try:
                coefs_arm = lin_arm.coef_ if not callable(lin_arm.coef_) else lin_arm.coef_()
                coefs_arm = np.array(coefs_arm).flatten()
            except Exception:
                coefs_arm = np.array(coefs).flatten()
            s = pd.Series(np.abs(coefs_arm), index=FEAT).sort_values(ascending=False).head(12)
            s.sort_values().plot(kind='barh', ax=ax, color='coral', alpha=0.8, edgecolor='white')
            ax.set_title(TREATMENT_LABELS[tc][:28], fontsize=9, fontweight='bold')
            ax.set_xlabel('|Coefficient|')
        plt.suptitle('LinearDML: Treatment Effect Modifiers per Arm', fontsize=11)
        plt.tight_layout()
        save_fig(fig, 'fig4_lineardml_coefs.png')
    except Exception as ex:
        log.warning(f'Fig4 skipped: {ex}')
    pfig.update(1)

    # ── Fig 5: CausalForest individual CI ────────────────────────────────────
    fig, axes = plt.subplots(1, min(2, n_arms), figsize=(13, 5), squeeze=False)
    axes = axes[0]
    for ax, tc in zip(axes, T_cols[:2]):
        if tc not in all_ci['CausalForestDML']:
            ax.axis('off'); continue
        ite = all_ite['CausalForestDML'][tc]
        lo, hi = all_ci['CausalForestDML'][tc]
        order  = np.argsort(ite)[:100]   # show 100 patients sorted by ITE
        pts    = np.arange(len(order))
        ax.scatter(pts, ite[order], s=8, color='#DD8452', zorder=3,
                   label='ITE point estimate')
        ax.fill_between(pts, lo[order], hi[order], alpha=0.25,
                        color='#DD8452', label='95% CI')
        ax.axhline(0, color='gray', linestyle='--', lw=1.5)
        sig_pct = metrics_all['CausalForestDML'][tc].get('pct_sig_ITE', np.nan)
        ax.set_title(f'CausalForestDML — {TREATMENT_LABELS[tc][:28]}\n'
                     f'({sig_pct:.1f}% patients with CI ≠ 0)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Patient (sorted by ITE)')
        ax.set_ylabel('ITE with 95% CI')
        ax.legend(fontsize=9)
    plt.suptitle('Individual Treatment Effect Uncertainty (CausalForestDML)', fontsize=11)
    plt.tight_layout()
    save_fig(fig, 'fig5_cf_individual_ci.png')
    pfig.update(1)

    # ── Fig 6: Policy value comparison ───────────────────────────────────────
    fig, axes = plt.subplots(1, n_arms, figsize=(5*n_arms+1, 5), squeeze=False)
    axes = axes[0]
    for ax, tc in zip(axes, T_cols):
        strategies = ['ITE Policy', 'Treat All', 'Treat None']
        vals_lin = [metrics_all['LinearDML'][tc]['policy'],
                    metrics_all['LinearDML'][tc]['treat_all'],
                    metrics_all['LinearDML'][tc]['treat_none']]
        vals_cf  = [metrics_all['CausalForestDML'][tc]['policy'],
                    metrics_all['CausalForestDML'][tc]['treat_all'],
                    metrics_all['CausalForestDML'][tc]['treat_none']]
        x_pos = np.arange(3)
        ax.bar(x_pos - 0.2, vals_lin, 0.35, label='LinearDML',
               color=COLORS['LinearDML'], alpha=0.85, edgecolor='white')
        ax.bar(x_pos + 0.2, vals_cf,  0.35, label='CausalForestDML',
               color=COLORS['CausalForestDML'], alpha=0.85, edgecolor='white')
        ax.set_xticks(x_pos); ax.set_xticklabels(strategies, fontsize=9)
        ax.set_ylabel('Mortality rate (lower=better)')
        ax.set_title(TREATMENT_LABELS[tc][:28], fontsize=9, fontweight='bold')
        ax.legend(fontsize=8)
    plt.suptitle('Policy Value: ITE-guided vs Treat-All vs Treat-None\n'
                 '(lower mortality = better policy)', fontsize=11)
    plt.tight_layout()
    save_fig(fig, 'fig6_policy_value.png')
    pfig.update(1)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE PER-PATIENT ITE SCORES
# ═══════════════════════════════════════════════════════════════════════════════
scores_df = pd.DataFrame({'patient_id': ite_df['patient_id'].values,
                           'Y': Y, 'T_any': T_bin})
for mname in all_ite:
    for tc in T_cols:
        col = f'ITE_{tc}_{mname.replace(" ","_")}'
        scores_df[col] = all_ite[mname][tc]

scores_df.to_csv(OUTPUT_DIR / 'ite_scores_final.csv', index=False)

export = {}
for mname in all_ite:
    export[mname] = {}
    for tc in T_cols:
        m = metrics_all[mname][tc]
        export[mname][tc] = {k: float(v) for k, v in m.items()
                              if isinstance(v, (float, int, np.floating, np.integer))}
with open(OUTPUT_DIR / 'final_results.json', 'w') as f:
    json.dump(export, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  COMPLETE")
print("="*65)
print(f"  Backend     : scikit-learn CPU (add cuML for GPU)")
print(f"  CV folds    : {CV_FOLDS}")
print(f"  Bootstrap   : {N_BOOT} iterations")
print(f"  Arms        : {len(T_cols)}")
print(f"  Models      : LinearDML + CausalForestDML")
print()
print(f"  {'Arm':<28} {'Model':<18} {'ATE':>8}  {'AUUC':>8}  "
      f"{'Qini':>8}  {'PG':>8}  {'Sig':>4}")
print(f"  {'-'*28} {'-'*18} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*4}")
for tc in T_cols:
    for mname in all_ite:
        m   = metrics_all[mname][tc]
        sig = '✓' if not (m['ATE_lo']<0<m['ATE_hi']) else '○'
        pg  = m['treat_none'] - m['policy']
        print(f"  {TREATMENT_LABELS[tc][:28]:<28} {mname:<18} "
              f"{m['ATE']:>+8.4f}  {m['AUUC']:>8.4f}  "
              f"{m['Qini']:>8.4f}  {pg:>+8.4f}  {sig:>4}")
print()
print("  OUTPUT FILES:")
for fp in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {fp.name}")
print()
print("  THESIS REPORTING:")
print("  Primary result  → CausalForestDML (honest CI, cite Wager&Athey 2018)")
print("  Interpretation  → LinearDML coefficients (which features modify effect)")
print("  Sensitivity     → compare ATE/AUUC/Qini between both models")
print("  Key table       → thesis_table_wide.csv")
print("="*65)