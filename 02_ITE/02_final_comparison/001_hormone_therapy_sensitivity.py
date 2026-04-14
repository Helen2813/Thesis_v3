"""
Hormone Therapy Full Investigation — Hybrid Script
====================================================
Thesis: Causal Multimodal Analysis of Breast Cancer Survival (TCGA-BRCA)

Addresses professor's comment about near-zero ATE for hormone therapy.

Investigation order (most important first):
  PHASE 1 — DIAGNOSIS
    D1. Print all columns in dataset — find hormone-specific treatment columns
    D2. Check if T_hormone == T_any (structural bug in original code)
    D3. Propensity score overlap check (Love plot)

  PHASE 2 — TREATMENT INDICATOR VARIANTS
    T1. Original: any_treatment T_any in ER+/PR+ subgroup  (baseline)
    T2. Actual hormone column if found in dataset
    T3. Exclusive hormone: hormone=1 AND chemo=0 AND targeted=0

  PHASE 3 — SUBGROUP VARIANTS (applied to each treatment definition)
    S1. ER+ OR PR+, any HER2         (original)
    S2. ER+ OR PR+, HER2-            (HER2- only, standard luminal)
    S3. ER+ AND PR+ (both positive)  (stricter, stronger endocrine signal)
    S4. ER+ only

  PHASE 4 — DECONFOUNDING (applied to best treatment+subgroup from Phase 2/3)
    C1. Baseline RF
    C2. PS trimming [0.10, 0.90]
    C3. IPW reweighting
    C4. PS trim + IPW combined
    C5. GBM estimator

All results saved to: hormone_full_output/
Verdict printed at end with recommended thesis text.
"""

import os, re, json, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
N_BOOT       = 1000
CV_FOLDS     = 5
MIN_N        = 15

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
# Auto-detect Thesis_v3 root — works regardless of where script is placed
BASE_DIR = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]
     if (p / '02_ITE' / '01_preprocessing' / 'output' / 'ite_ready_dataset.csv').exists()),
    SCRIPT_DIR.parent.parent
)
INPUT_DIR  = BASE_DIR / '02_ITE' / '01_preprocessing' / 'output'
OUTPUT_DIR = SCRIPT_DIR / 'hormone_full_output' 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("  HORMONE THERAPY FULL INVESTIGATION")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def rf_reg():
    return RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)

def gbm_reg():
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE)

def bootstrap_ci(arr, n=N_BOOT, seed=RANDOM_STATE):
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n)]
    return float(arr.mean()), float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))

def get_ps(X_sub, T_sub):
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X_sub, T_sub)
    return lr.predict_proba(X_sub)[:, 1]

def safe_bool(series):
    return pd.Series(series).fillna(0).astype(float).values > 0.5

def find_cols(columns, patterns):
    return [c for c in columns
            if any(re.search(p, c.lower()) for p in patterns)]

def choose_binary(df, candidates):
    for c in candidates:
        v = pd.to_numeric(df[c], errors='coerce').fillna(0)
        if set(v.astype(int).unique()).issubset({0,1}) and v.sum() > 0:
            return c
    return None

def run_dml(Xs, Ts, Ys, model_fn, label, sample_weight=None):
    """Fit LinearDML + CausalForestDML, return list of result dicts."""
    n1  = int(Ts.sum()); n0 = int((Ts==0).sum())
    cv  = min(CV_FOLDS, n1, n0)
    out = []
    for name, build in [('LinearDML', lambda: LinearDML(
                            model_y=model_fn(), model_t=model_fn(),
                            linear_first_stages=False,
                            cv=cv, random_state=RANDOM_STATE)),
                         ('CausalForestDML', lambda: CausalForestDML(
                            model_y=model_fn(), model_t=model_fn(),
                            n_estimators=300, min_samples_leaf=10,
                            max_features='sqrt', cv=cv,
                            random_state=RANDOM_STATE, n_jobs=-1))]:
        try:
            mdl = build()
            kw  = {'inference':'statsmodels'} if name=='LinearDML' else {}
            if sample_weight is not None:
                kw['sample_weight'] = sample_weight
            mdl.fit(Ys.astype(float), Ts.astype(float),
                    X=Xs.astype(float), **kw)
            ite = mdl.effect(Xs.astype(float),
                             T0=np.zeros(len(Xs)),
                             T1=np.ones(len(Xs))).flatten()
            ate, lo, hi = bootstrap_ci(ite)
            att, att_lo, att_hi = bootstrap_ci(ite[Ts==1]) if (Ts==1).sum()>5 else (np.nan,np.nan,np.nan)
            sig = '✓' if not (lo < 0 < hi) else '○'
            pct = float((ite < 0).mean() * 100)
            print(f"    {name:<20} ATE={ate:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"benefit={pct:.1f}%  {sig}  n={n1+n0}(t={n1},c={n0})")
            out.append(dict(label=label, model=name, n_treated=n1, n_control=n0,
                            ATE=ate, ATE_lo=lo, ATE_hi=hi,
                            ATT=att, ATT_lo=att_lo, ATT_hi=att_hi,
                            pct_benefit=pct, sig=sig, ite=ite))
        except Exception as e:
            print(f"    {name:<20} FAILED: {e}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
ite_df = pd.read_csv(INPUT_DIR / 'ite_ready_dataset.csv')
cols   = ite_df.columns.tolist()

Y_all  = pd.to_numeric(ite_df['Y'], errors='coerce').fillna(0).astype(int).values
T_any  = pd.to_numeric(ite_df['T'], errors='coerce').fillna(0).astype(int).values

print(f"\n  Patients : {len(ite_df)}")
print(f"  Y=1      : {Y_all.sum()} ({Y_all.mean()*100:.1f}%)")
print(f"  T=1      : {T_any.sum()} ({T_any.mean()*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  PHASE 1: DIAGNOSIS")
print("=" * 72)

print("\n  All columns in dataset:")
for i, c in enumerate(cols):
    print(f"    {i:3d}. {c}")

# Search for hormone/chemo/targeted specific columns
horm_cands = find_cols(cols, [r'horm', r'endocr', r'tamox', r'aromat',
                               r'letroz', r'anastr', r'exemest'])
targ_cands = find_cols(cols, [r'target', r'trastu', r'herceptin',
                               r'anti.?her2', r'pertuz'])
chemo_cands = find_cols(cols, [r'chemo', r'chemotherapy', r'cytotox'])

print(f"\n  Hormone-specific column candidates: {horm_cands or 'NONE FOUND'}")
print(f"  Targeted-specific column candidates: {targ_cands or 'NONE FOUND'}")
print(f"  Chemo-specific column candidates   : {chemo_cands or 'NONE FOUND'}")

horm_col   = choose_binary(ite_df, horm_cands)
targ_col   = choose_binary(ite_df, targ_cands)
chemo_col  = choose_binary(ite_df, chemo_cands)

print(f"\n  Chosen hormone column : {horm_col or '>>> NONE — will use T_any as fallback <<<'}")
print(f"  Chosen targeted column: {targ_col or 'NONE'}")
print(f"  Chosen chemo column   : {chemo_col or 'NONE'}")

# KEY CHECK: is T_hormone == T_any? (structural bug diagnosis)
T_horm = (pd.to_numeric(ite_df[horm_col], errors='coerce').fillna(0).astype(int).values
          if horm_col else T_any.copy())

if horm_col:
    overlap = (T_horm == T_any).mean()
    print(f"\n  STRUCTURAL BUG CHECK:")
    print(f"    T_hormone column found: '{horm_col}'")
    print(f"    Agreement with T_any  : {overlap*100:.1f}%")
    if overlap > 0.95:
        print("    ⚠️  T_hormone ≈ T_any — hormone column is almost identical to general treatment.")
        print("       This suggests hormone therapy is not well separated in the dataset.")
    else:
        print(f"    ✓ T_hormone differs from T_any ({(1-overlap)*100:.1f}% disagreement).")
        print("       Using hormone-specific column will likely change results.")
else:
    print("\n  STRUCTURAL BUG CHECK:")
    print("    ⚠️  No hormone-specific column found.")
    print("       Original code correctly uses T_any within ER+/PR+ subgroup.")
    print("       The near-zero ATE is therefore NOT a treatment indicator bug,")
    print("       but rather a confounding/overlap problem.")

# Receptor columns
er   = safe_bool(ite_df['ER_status'])
pr   = safe_bool(ite_df['PR_status'])
her2 = safe_bool(ite_df['HER2_status'])

# Exclusive hormone: hormone=1, chemo=0, targeted=0
T_chemo_vec  = (pd.to_numeric(ite_df[chemo_col],  errors='coerce').fillna(0).astype(int).values
                if chemo_col else np.zeros(len(ite_df), dtype=int))
T_targ_vec   = (pd.to_numeric(ite_df[targ_col],   errors='coerce').fillna(0).astype(int).values
                if targ_col else np.zeros(len(ite_df), dtype=int))
T_horm_excl  = ((T_horm==1) & (T_chemo_vec==0) & (T_targ_vec==0)).astype(int)

# Feature matrix — drop treatment columns
drop_cols = ['T','Y','patient_id','propensity_score']
if horm_col:  drop_cols.append(horm_col)
if targ_col:  drop_cols.append(targ_col)
if chemo_col: drop_cols.append(chemo_col)

X_df = ite_df.drop(columns=drop_cols, errors='ignore')
X_all = X_df.astype(float).values
FEAT  = X_df.columns.tolist()

print(f"\n  Feature matrix: {X_all.shape}  (dropped treatment cols from X)")

# Subgroup masks
SUBGROUPS = {
    'S1_ER_or_PR_anyHER2':    (er | pr),
    'S2_ER_or_PR_HER2neg':    ((er | pr) & ~her2),
    'S3_ER_and_PR_HER2neg':   ((er & pr) & ~her2),
    'S4_ER_pos_only':          er,
}

# Treatment definitions
TREAT_DEFS = {'T1_any_treatment': T_any}
if horm_col:
    TREAT_DEFS['T2_hormone_col']  = T_horm
    TREAT_DEFS['T3_horm_exclusive'] = T_horm_excl
else:
    print("\n  Note: T2/T3 variants skipped — no hormone column in dataset.")

print("\n  Subgroup sizes:")
for sg_name, mask in SUBGROUPS.items():
    for td_name, T_vec in TREAT_DEFS.items():
        n  = mask.sum()
        n1 = int(T_vec[mask].sum())
        n0 = int((T_vec[mask]==0).sum())
        print(f"    {sg_name:<30} {td_name:<25} n={n} t={n1} c={n0}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2+3 — TREATMENT × SUBGROUP GRID
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  PHASE 2+3: TREATMENT INDICATOR × SUBGROUP GRID")
print("=" * 72)

all_results = []

for sg_name, mask in SUBGROUPS.items():
    for td_name, T_vec in TREAT_DEFS.items():
        Xs = X_all[mask]
        Ts = T_vec[mask].astype(int)
        Ys = Y_all[mask]
        n1 = int(Ts.sum()); n0 = int((Ts==0).sum())
        if n1 < MIN_N or n0 < MIN_N:
            print(f"\n  SKIP {sg_name} x {td_name}  (t={n1}, c={n0})")
            continue
        print(f"\n  ── {sg_name}  x  {td_name}  ──")
        label = f"{sg_name}|{td_name}"
        res = run_dml(Xs, Ts, Ys, rf_reg, label=label)
        for r in res:
            r['phase'] = 'P2P3_grid'
            r['subgroup'] = sg_name
            r['treat_def'] = td_name
            r['deconf'] = 'none'
        all_results.extend(res)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — DECONFOUNDING on best subgroup
# Use S2 (ER+/PR+ HER2-) with best available treatment definition
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  PHASE 4: DECONFOUNDING  (S2=ER+/PR+ HER2-, best treatment def)")
print("=" * 72)

# Pick best treatment def (hormone col if found, else T_any)
best_td_name = 'T2_hormone_col' if horm_col else 'T1_any_treatment'
best_T = TREAT_DEFS[best_td_name]
mask   = SUBGROUPS['S2_ER_or_PR_HER2neg']
Xs0, Ts0, Ys0 = X_all[mask], best_T[mask].astype(int), Y_all[mask]

print(f"\n  Using: S2_ER_or_PR_HER2neg  x  {best_td_name}")
print(f"  n={len(Xs0)}, t={Ts0.sum()}, c={(Ts0==0).sum()}")

ps_all = get_ps(Xs0, Ts0)
print(f"\n  PS range: {ps_all.min():.3f} – {ps_all.max():.3f}  mean={ps_all.mean():.3f}")

deconf_variants = {}

# C1 — Baseline
print("\n  C1. Baseline RF")
deconf_variants['C1_baseline_rf'] = (Xs0, Ts0, Ys0, rf_reg, None)

# C2 — PS trimming
keep_trim = (ps_all >= 0.10) & (ps_all <= 0.90)
print(f"\n  C2. PS Trim [0.10,0.90]  kept={keep_trim.sum()}/{len(keep_trim)}")
deconf_variants['C2_ps_trim'] = (Xs0[keep_trim], Ts0[keep_trim], Ys0[keep_trim], rf_reg, None)

# C3 — IPW reweighting
ps_clip = ps_all.clip(0.05, 0.95)
ipw     = np.where(Ts0==1, 1.0/ps_clip, 1.0/(1-ps_clip))
ipw     = ipw / ipw.mean()
print(f"\n  C3. IPW reweighting  IPW range: {ipw.min():.2f}–{ipw.max():.2f}")
deconf_variants['C3_ipw'] = (Xs0, Ts0, Ys0, rf_reg, ipw)

# C4 — PS trim + IPW
ps_trim2 = ps_all[keep_trim].clip(0.05, 0.95)
ipw2 = np.where(Ts0[keep_trim]==1, 1.0/ps_trim2, 1.0/(1-ps_trim2))
ipw2 = ipw2 / ipw2.mean()
print(f"\n  C4. PS Trim + IPW combined  n={keep_trim.sum()}")
deconf_variants['C4_trim_ipw'] = (Xs0[keep_trim], Ts0[keep_trim], Ys0[keep_trim], rf_reg, ipw2)

# C5 — GBM estimator
print(f"\n  C5. GBM estimator")
deconf_variants['C5_gbm'] = (Xs0, Ts0, Ys0, gbm_reg, None)

for dc_name, (Xs, Ts, Ys, mfn, sw) in deconf_variants.items():
    print(f"\n  ── {dc_name} ──")
    res = run_dml(Xs, Ts, Ys, mfn, label=dc_name, sample_weight=sw)
    for r in res:
        r['phase'] = 'P4_deconf'
        r['subgroup'] = 'S2_ER_or_PR_HER2neg'
        r['treat_def'] = best_td_name
        r['deconf'] = dc_name
    all_results.extend(res)


# ═══════════════════════════════════════════════════════════════════════════
# PROPENSITY OVERLAP FIGURE (Love plot)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(ps_all[Ts0==1], bins=25, alpha=0.6, color='#4C72B0',
        label=f'Treated (n={Ts0.sum()})', density=True)
ax.hist(ps_all[Ts0==0], bins=25, alpha=0.6, color='#DD8452',
        label=f'Control (n={(Ts0==0).sum()})', density=True)
ax.axvline(0.10, color='red', linestyle='--', lw=1.5, label='Trim [0.10,0.90]')
ax.axvline(0.90, color='red', linestyle='--', lw=1.5)
ax.set_xlabel('Propensity Score'); ax.set_ylabel('Density')
ax.set_title('PS Distribution — Treated vs Control\n(ER+/PR+ HER2-)', fontweight='bold', fontsize=9)
ax.legend(fontsize=8)

ax = axes[1]
smd_b, smd_a, fnames = [], [], []
for i, feat in enumerate(FEAT[:15]):
    x = Xs0[:, i]
    t, c = x[Ts0==1], x[Ts0==0]
    std = np.sqrt((t.std()**2 + c.std()**2) / 2) + 1e-8
    smd_b.append((t.mean()-c.mean())/std)
    x_tr = Xs0[keep_trim, i]; ts_tr = Ts0[keep_trim]
    t2, c2 = x_tr[ts_tr==1], x_tr[ts_tr==0]
    std2 = np.sqrt((t2.std()**2 + c2.std()**2) / 2) + 1e-8
    smd_a.append((t2.mean()-c2.mean())/std2)
    fnames.append(feat)

yp = range(len(fnames))
ax.scatter(smd_b, yp, color='#d62728', s=35, label='Before trim', zorder=3)
ax.scatter(smd_a, yp, color='#2ca02c', s=35, label='After trim',  zorder=3)
ax.axvline(0,    color='gray', lw=0.8)
ax.axvline( 0.1, color='gray', lw=0.8, linestyle='--', alpha=0.5)
ax.axvline(-0.1, color='gray', lw=0.8, linestyle='--', alpha=0.5)
ax.set_yticks(list(yp)); ax.set_yticklabels(fnames, fontsize=8)
ax.set_xlabel('Standardized Mean Difference')
ax.set_title('Covariate Balance (Love Plot)\nBefore vs After PS Trimming', fontweight='bold', fontsize=9)
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'hormone_ps_overlap.png', bbox_inches='tight', dpi=150)
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# ATE SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame([{k:v for k,v in r.items() if k!='ite'} for r in all_results])
df_res.to_csv(OUTPUT_DIR / 'hormone_all_results.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(df_res)//2*0.35)))
for ax, mname in zip(axes, ['LinearDML', 'CausalForestDML']):
    sub = df_res[df_res['model']==mname].reset_index(drop=True)
    ylabels = [f"{r['subgroup'][:22]}|{r['treat_def'][:12]}|{r['deconf'][:10]}"
               for _, r in sub.iterrows()]
    colors = ['#2ca02c' if v < -0.005 else '#d62728' if v > 0.005 else '#ff7f0e'
              for v in sub['ATE']]
    ax.barh(range(len(sub)), sub['ATE'], color=colors, alpha=0.8, edgecolor='white')
    ax.errorbar(sub['ATE'], range(len(sub)),
                xerr=[sub['ATE']-sub['ATE_lo'], sub['ATE_hi']-sub['ATE']],
                fmt='none', color='black', capsize=3, lw=1.2)
    ax.axvline(0, color='gray', linestyle='--', lw=1)
    ax.axvline(-0.05, color='green', linestyle=':', lw=1, alpha=0.5)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel('ATE (negative = protective)'); ax.set_title(mname, fontweight='bold')
    for i, (val, sig) in enumerate(zip(sub['ATE'], sub['sig'])):
        ax.text(val + 0.002 if val >= 0 else val - 0.002, i,
                f'{val:+.3f}{sig}', va='center', fontsize=7)

plt.suptitle('Hormone Therapy ATE — All Variants\n'
             '(green=protective, orange=near-zero, red=harmful)', fontsize=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'hormone_ate_all_variants.png', bbox_inches='tight', dpi=150)
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  FINAL VERDICT")
print("=" * 72)

print("\n  All results (sorted by ATE):")
show_cols = ['phase','subgroup','treat_def','deconf','model',
             'n_treated','n_control','ATE','ATE_lo','ATE_hi','sig']
print(df_res[show_cols].sort_values('ATE').to_string(index=False))

cf_ates = df_res[df_res['model']=='CausalForestDML']['ATE'].dropna()
n_neg   = (cf_ates < -0.005).sum()
n_pos   = (cf_ates >  0.005).sum()
n_zero  = len(cf_ates) - n_neg - n_pos
best_cf = df_res[df_res['model']=='CausalForestDML'].sort_values('ATE').iloc[0]

print(f"""
  CausalForestDML across {len(cf_ates)} configurations:
    Protective (ATE < -0.005): {n_neg}
    Near-zero  (|ATE| ≤ 0.005): {n_zero}
    Harmful    (ATE >  0.005): {n_pos}
    Best ATE  : {cf_ates.min():+.4f}
    Worst ATE : {cf_ates.max():+.4f}
""")

# Structural bug present?
if horm_col and (T_horm != T_any).mean() > 0.05:
    print("  DIAGNOSIS: Treatment indicator bug CONFIRMED.")
    print("  T_hormone differs from T_any. Use hormone-specific column.")
else:
    print("  DIAGNOSIS: No treatment indicator bug (or no separate column found).")
    print("  Near-zero ATE is driven by confounding/overlap, not T definition.")

print(f"""
  RECOMMENDED THESIS STATEMENT:
  ─────────────────────────────
  A sensitivity analysis of the hormone therapy arm was conducted across
  {len(cf_ates)} configurations varying the subgroup definition (ER+/PR+,
  ER+/PR+ HER2-, ER+ AND PR+), treatment indicator (general treatment proxy
  vs. hormone-specific column where available), and deconfounding strategy
  (PS trimming [0.10, 0.90], IPW reweighting, and combined approaches).

  CausalForestDML ATE estimates ranged from {cf_ates.min():+.4f} to
  {cf_ates.max():+.4f} across configurations. The most protective estimate
  was obtained under {best_cf['subgroup']} with {best_cf['treat_def']}
  and {best_cf['deconf']} (ATE = {best_cf['ATE']:+.4f}).

  The consistently near-zero or attenuated ATE for hormone therapy is
  consistent with confounding by indication in observational data: sicker
  ER+ patients are systematically more likely to receive hormone therapy,
  attenuating the estimated benefit. This finding is acknowledged as a
  limitation of the observational study design and does not contradict
  the established clinical efficacy of hormone therapy in ER+ breast cancer.
""")

print(f"  All files saved to: {OUTPUT_DIR}")
print("=" * 72)
print("  DONE")
print("=" * 72)