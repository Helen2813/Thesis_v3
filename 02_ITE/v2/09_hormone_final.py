"""
Hormone Therapy — Final Reliable Analysis
==========================================
Fixes vs previous scripts:
  1. Classifier for treatment model (binary T, not regressor)
  2. Repeated seeds instead of bootstrap-on-precomputed-ITE
  3. Prespecified primary + sensitivity subgroups (no post-hoc selection)
  4. Stabilized IPW
  5. TNBC sanity check
"""

import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from econml.dml import LinearDML, CausalForestDML
from econml.dr import LinearDRLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
SEEDS        = [42, 123, 456, 789, 1337, 2024, 31, 99, 7, 404]   # 10 seeds
CV_FOLDS     = 5
MIN_N        = 20

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR   = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]
     if (p / '02_ITE' / '01_preprocessing' / 'output' / 'ite_ready_dataset_v2.csv').exists()),
    None
)
if BASE_DIR is None:
    print("ERROR: ite_ready_dataset_v2.csv not found — run 04_create_treatment_arms.py first")
    sys.exit(1)

INPUT_DIR  = BASE_DIR / '02_ITE' / '01_preprocessing' / 'output'
OUTPUT_DIR = SCRIPT_DIR / 'hormone_final_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  HORMONE THERAPY — FINAL RELIABLE ANALYSIS")
print("=" * 65)

df = pd.read_csv(INPUT_DIR / 'ite_ready_dataset_v2.csv')
print(f"  Patients : {len(df)}")
print(f"  Y=1      : {df['Y'].sum()} ({df['Y'].mean()*100:.1f}%)")
print(f"  T_hormone: {df['T_hormone'].sum()} ({df['T_hormone'].mean()*100:.1f}%)")
print(f"  T_horm_excl: {df['T_hormone_excl'].sum()} ({df['T_hormone_excl'].mean()*100:.1f}%)")

X_all     = df.drop(columns=['T','Y','patient_id','propensity_score',
                              'T_hormone','T_chemo','T_targeted',
                              'T_radiation','T_hormone_excl'], errors='ignore').values.astype(float)
FEAT      = df.drop(columns=['T','Y','patient_id','propensity_score',
                              'T_hormone','T_chemo','T_targeted',
                              'T_radiation','T_hormone_excl'], errors='ignore').columns.tolist()
Y_all     = df['Y'].astype(int).values
er        = (df['ER_status'].fillna(0) > 0.5).astype(int).values
pr        = (df['PR_status'].fillna(0) > 0.5).astype(int).values
her2      = (df['HER2_status'].fillna(0) > 0.5).astype(int).values


# ── Model builders — classifier for treatment ────────────────────────────────

def rf_reg(seed=RANDOM_STATE):
    return RandomForestRegressor(n_estimators=200, max_depth=6,
                                 min_samples_leaf=10, max_features='sqrt',
                                 n_jobs=-1, random_state=seed)

def rf_clf(seed=RANDOM_STATE):
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=10, max_features='sqrt',
                                  class_weight='balanced', n_jobs=-1, random_state=seed)


# ── Stabilized IPW ────────────────────────────────────────────────────────────

def stabilized_ipw(X, T, clip=(0.05, 0.95)):
    lr  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X, T)
    ps  = lr.predict_proba(X)[:, 1].clip(*clip)
    p1  = T.mean()
    sw  = np.where(T == 1, p1 / ps, (1 - p1) / (1 - ps))
    return sw / sw.mean()


# ── Run DML with repeated seeds ───────────────────────────────────────────────

def run_repeated(Xs, Ts, Ys, label, sample_weight=None):
    n1 = int(Ts.sum()); n0 = int((Ts == 0).sum())
    if n1 < MIN_N or n0 < MIN_N:
        print(f"    SKIP {label}: n1={n1} n0={n0}")
        return []

    lin_ates, cf_ates, dr_ates = [], [], []

    for seed in SEEDS:
        try:
            cv = min(CV_FOLDS, n1, n0)

            lin = LinearDML(model_y=rf_reg(seed), model_t=rf_clf(seed),
                            discrete_treatment=True,
                            linear_first_stages=False,
                            cv=cv, random_state=seed)
            kw = {'inference': 'statsmodels'}
            if sample_weight is not None: kw['sample_weight'] = sample_weight
            lin.fit(Ys.astype(float), Ts, X=Xs.astype(float), **kw)
            lin_ates.append(float(lin.effect(Xs.astype(float)).mean()))

            cf = CausalForestDML(model_y=rf_reg(seed), model_t=rf_clf(seed),
                                 discrete_treatment=True,
                                 n_estimators=200, min_samples_leaf=10,
                                 max_features='sqrt', cv=cv,
                                 random_state=seed, n_jobs=-1)
            if sample_weight is not None:
                cf.fit(Ys.astype(float), Ts, X=Xs.astype(float), sample_weight=sample_weight)
            else:
                cf.fit(Ys.astype(float), Ts, X=Xs.astype(float))
            cf_ates.append(float(cf.effect(Xs.astype(float)).mean()))

            # DR-learner: doubly robust, different identification strategy
            dr = LinearDRLearner(model_regression=rf_reg(seed),
                                 model_propensity=rf_clf(seed),
                                 cv=cv, random_state=seed)
            dr_kw = {}
            if sample_weight is not None: dr_kw['sample_weight'] = sample_weight
            dr.fit(Ys.astype(float), Ts, X=Xs.astype(float), **dr_kw)
            dr_ates.append(float(dr.effect(Xs.astype(float)).mean()))

        except Exception as e:
            print(f"    seed={seed} FAILED: {e}")

    def summarize(ates, name):
        if not ates:
            return {}
        ates_arr  = np.array(ates)
        med       = float(np.median(ates_arr))
        iqr_lo    = float(np.percentile(ates_arr, 25))
        iqr_hi    = float(np.percentile(ates_arr, 75))
        mn, mx    = float(ates_arr.min()), float(ates_arr.max())
        sign_pct  = float((ates_arr < 0).mean() * 100)
        consistent = sign_pct >= 80 or sign_pct <= 20
        direction  = 'protective' if med < 0 else 'harmful' if med > 0 else 'neutral'
        stable     = '✓ stable' if consistent else '~ unstable'
        n_seeds    = len(ates)
        print(f"    {name:<20} median={med:+.4f}  IQR=[{iqr_lo:+.4f},{iqr_hi:+.4f}]  "
              f"range=[{mn:+.4f},{mx:+.4f}]  protective={sign_pct:.0f}%/{n_seeds}  {stable}")
        return dict(label=label, model=name, n_treated=n1, n_control=n0,
                    median_ATE=med, IQR_lo=iqr_lo, IQR_hi=iqr_hi,
                    ATE_min=mn, ATE_max=mx,
                    pct_protective=sign_pct, stable=consistent,
                    direction=direction, all_ates=ates)

    results = []
    lin_r = summarize(lin_ates, 'LinearDML')
    cf_r  = summarize(cf_ates,  'CausalForestDML')
    dr_r  = summarize(dr_ates,  'LinearDRLearner')
    if lin_r: results.append(lin_r)
    if cf_r:  results.append(cf_r)
    if dr_r:  results.append(dr_r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PRESPECIFIED ANALYSIS PLAN
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  PRESPECIFIED ANALYSIS PLAN:")
print("  Primary  : S2 (ER+/PR+ HER2-) × T_hormone")
print("  Sensitivity 1: S3 (ER+ AND PR+ HER2-) × T_hormone")
print("  Sensitivity 2: S4 (ER+ only) × T_hormone")
print("  Sensitivity 3: S2 × T_hormone_excl (exclusive, no chemo/targeted)")
print("  Sanity check : TNBC × T_hormone (should NOT be strongly protective)")
print()

all_results = []

ANALYSES = [
    ('PRIMARY_S2_horm',        (er | pr) & (her2 == 0), df['T_hormone'].values,      'Primary:  ER+/PR+ HER2-    × T_hormone'),
    ('SENS1_S3_horm',          (er & pr) & (her2 == 0), df['T_hormone'].values,      'Sens-1:   ER+ AND PR+ HER2-× T_hormone'),
    ('SENS2_S4_horm',          er,                       df['T_hormone'].values,      'Sens-2:   ER+ only         × T_hormone'),
    ('SENS3_S2_horm_excl',     (er | pr) & (her2 == 0), df['T_hormone_excl'].values, 'Sens-3:   ER+/PR+ HER2-    × T_hormone_excl'),
    ('SANITY_TNBC_horm',       (er == 0) & (pr == 0) & (her2 == 0), df['T_hormone'].values, 'SANITY:   TNBC              × T_hormone'),
]

for label, mask, T_vec, desc in ANALYSES:
    print(f"\n  ── {desc} ──")
    mask    = mask.astype(bool)
    Xs      = X_all[mask]
    Ts      = T_vec[mask].astype(int)
    Ys      = Y_all[mask]
    n1      = int(Ts.sum()); n0 = int((Ts == 0).sum())
    print(f"    n={mask.sum()}, treated={n1}, control={n0}")

    # Baseline
    res = run_repeated(Xs, Ts, Ys, label)
    for r in res:
        r['variant'] = 'baseline'; all_results.append(r)

    if n1 < MIN_N or n0 < MIN_N:
        continue

    # Stabilized IPW
    print(f"    [+ stabilized IPW]")
    sw = stabilized_ipw(Xs, Ts)
    res_ipw = run_repeated(Xs, Ts, Ys, label + '_sipw', sample_weight=sw)
    for r in res_ipw:
        r['variant'] = 'stabilized_ipw'; all_results.append(r)

    # PS trimming [0.10, 0.90]
    lr_ps = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr_ps.fit(Xs, Ts)
    ps    = lr_ps.predict_proba(Xs)[:, 1]
    keep  = (ps >= 0.10) & (ps <= 0.90)
    print(f"    [+ PS trim 0.10–0.90]  kept={keep.sum()}/{len(keep)}")
    if keep.sum() > 40:
        res_trim = run_repeated(Xs[keep], Ts[keep], Ys[keep], label + '_trim')
        for r in res_trim:
            r['variant'] = 'ps_trim_10_90'; all_results.append(r)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)

df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'all_ates'}
                        for r in all_results])
df_res.to_csv(OUTPUT_DIR / 'hormone_final_results.csv', index=False)

primary = df_res[df_res['label'].str.startswith('PRIMARY')]
sanity  = df_res[df_res['label'].str.startswith('SANITY')]

print("\n  PRIMARY (ER+/PR+ HER2- × T_hormone):")
print(primary[['model','variant','n_treated','n_control',
               'median_ATE','IQR_lo','IQR_hi','ATE_min','ATE_max',
               'pct_protective','stable','direction']].to_string(index=False))

print("\n  SENSITIVITY analyses:")
sens = df_res[df_res['label'].str.startswith('SENS')]
print(sens[['label','model','variant','median_ATE','pct_protective','stable']].to_string(index=False))

print("\n  SANITY CHECK (TNBC × T_hormone — should NOT be strongly protective):")
print(sanity[['model','variant','median_ATE','pct_protective','stable']].to_string(index=False))

# Two-model agreement on sign in primary
if len(primary) > 0:
    prot = (primary['pct_protective'] >= 60).mean()
    print(f"\n  Primary: {prot*100:.0f}% of model/variant combos show protective direction (>=60%)")


# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mname in zip(axes, ['LinearDML', 'CausalForestDML']):
    sub = df_res[df_res['model'] == mname].copy()
    sub['y_label'] = sub['label'] + ' | ' + sub['variant']
    colors = ['#2ca02c' if v < -0.003 else '#ff7f0e' if abs(v) <= 0.003 else '#d62728'
              for v in sub['median_ATE']]
    ax.barh(range(len(sub)), sub['median_ATE'], color=colors, alpha=0.8)
    ax.errorbar(sub['median_ATE'], range(len(sub)),
                xerr=[sub['median_ATE'] - sub['IQR_lo'],
                      sub['IQR_hi'] - sub['median_ATE']],
                fmt='none', color='black', capsize=3, lw=1.2)
    ax.axvline(0, color='gray', linestyle='--', lw=1)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub['y_label'], fontsize=7)
    ax.set_xlabel('Median ATE across 5 seeds (IQR bars)')
    ax.set_title(f'{mname}\n(discrete treatment, repeated seeds)', fontweight='bold', fontsize=9)

plt.suptitle('Hormone Therapy — Prespecified Analysis\n'
             '(green=protective, orange=near-zero, red=harmful)', fontsize=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'hormone_final_figure.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"\n  Saved: {OUTPUT_DIR}")
print("=" * 65)
print("  DONE")
print("=" * 65)