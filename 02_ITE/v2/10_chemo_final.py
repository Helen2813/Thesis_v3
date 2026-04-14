"""
Chemotherapy — Final Reliable Analysis
========================================
Same structure as 09_hormone_final.py.

Prespecified analysis plan:
  Primary    : TNBC (ER- PR- HER2-) × T_chemo
  Sensitivity 1: All patients × T_chemo
  Sensitivity 2: HER2- (any ER/PR) × T_chemo
  Sensitivity 3: TNBC × T_chemo_excl (chemo only, no targeted)
  Cross-subgroup: ER+/PR+ HER2- × T_chemo (not a falsification — chemo helps here too)
  Falsification: TNBC × T_targeted (anti-HER2 in HER2- tumors)
"""

import os, sys, warnings
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
SEEDS        = [42, 123, 456, 789, 1337, 2024, 31, 99, 7, 404]
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
OUTPUT_DIR = SCRIPT_DIR / 'chemo_final_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  CHEMOTHERAPY — FINAL RELIABLE ANALYSIS")
print("=" * 65)

df = pd.read_csv(INPUT_DIR / 'ite_ready_dataset_v2.csv')
print(f"  Patients : {len(df)}")
print(f"  Y=1      : {df['Y'].sum()} ({df['Y'].mean()*100:.1f}%)")
print(f"  T_chemo  : {df['T_chemo'].sum()} ({df['T_chemo'].mean()*100:.1f}%)")

drop_cols = ['T','Y','patient_id','propensity_score',
             'T_hormone','T_chemo','T_targeted','T_radiation','T_hormone_excl']
X_all = df.drop(columns=drop_cols, errors='ignore').values.astype(float)
Y_all = df['Y'].astype(int).values
er    = (df['ER_status'].fillna(0) > 0.5).astype(int).values
pr    = (df['PR_status'].fillna(0) > 0.5).astype(int).values
her2  = (df['HER2_status'].fillna(0) > 0.5).astype(int).values

T_chemo       = df['T_chemo'].astype(int).values
T_chemo_excl  = ((df['T_chemo'] == 1) & (df['T_targeted'] == 0) & (df['T_hormone'] == 0)).astype(int).values

print(f"  T_chemo_excl: {T_chemo_excl.sum()} ({T_chemo_excl.mean()*100:.1f}%)")


def rf_reg(seed=RANDOM_STATE):
    return RandomForestRegressor(n_estimators=200, max_depth=6,
                                 min_samples_leaf=10, max_features='sqrt',
                                 n_jobs=-1, random_state=seed)

def rf_clf(seed=RANDOM_STATE):
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=10, max_features='sqrt',
                                  class_weight='balanced', n_jobs=-1, random_state=seed)

def stabilized_ipw(X, T, clip=(0.05, 0.95)):
    lr  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X, T)
    ps  = lr.predict_proba(X)[:, 1].clip(*clip)
    p1  = T.mean()
    sw  = np.where(T == 1, p1 / ps, (1 - p1) / (1 - ps))
    return sw / sw.mean()

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
                            discrete_treatment=True, linear_first_stages=False,
                            cv=cv, random_state=seed)
            kw = {'inference': 'statsmodels'}
            if sample_weight is not None: kw['sample_weight'] = sample_weight
            lin.fit(Ys.astype(float), Ts, X=Xs.astype(float), **kw)
            lin_ates.append(float(lin.effect(Xs.astype(float)).mean()))

            cf = CausalForestDML(model_y=rf_reg(seed), model_t=rf_clf(seed),
                                 discrete_treatment=True, n_estimators=200,
                                 min_samples_leaf=10, max_features='sqrt',
                                 cv=cv, random_state=seed, n_jobs=-1)
            cf_kw = {} if sample_weight is None else {'sample_weight': sample_weight}
            cf.fit(Ys.astype(float), Ts, X=Xs.astype(float), **cf_kw)
            cf_ates.append(float(cf.effect(Xs.astype(float)).mean()))

            dr = LinearDRLearner(model_regression=rf_reg(seed),
                                 model_propensity=rf_clf(seed),
                                 cv=cv, random_state=seed)
            dr_kw = {} if sample_weight is None else {'sample_weight': sample_weight}
            dr.fit(Ys.astype(float), Ts, X=Xs.astype(float), **dr_kw)
            dr_ates.append(float(dr.effect(Xs.astype(float)).mean()))

        except Exception as e:
            print(f"    seed={seed} FAILED: {e}")

    def summarize(ates, name):
        if not ates:
            return {}
        a      = np.array(ates)
        med    = float(np.median(a))
        iqr_lo = float(np.percentile(a, 25))
        iqr_hi = float(np.percentile(a, 75))
        mn, mx = float(a.min()), float(a.max())
        prot   = float((a < 0).mean() * 100)
        stable = prot >= 80 or prot <= 20
        sym    = '✓ stable' if stable else '~ unstable'
        print(f"    {name:<20} median={med:+.4f}  IQR=[{iqr_lo:+.4f},{iqr_hi:+.4f}]  "
              f"range=[{mn:+.4f},{mx:+.4f}]  protective={prot:.0f}%/{len(ates)}  {sym}")
        return dict(label=label, model=name, n_treated=n1, n_control=n0,
                    median_ATE=med, IQR_lo=iqr_lo, IQR_hi=iqr_hi,
                    ATE_min=mn, ATE_max=mx, pct_protective=prot,
                    stable=stable, direction='protective' if med < 0 else 'harmful')

    results = []
    for ates, name in [(lin_ates,'LinearDML'),(cf_ates,'CausalForestDML'),(dr_ates,'LinearDRLearner')]:
        r = summarize(ates, name)
        if r: results.append(r)
    return results


# ── Subgroup sizes ────────────────────────────────────────────────────────────
tnbc     = ((er==0) & (pr==0) & (her2==0))
her2neg  = (her2==0)
luminal  = ((er|pr) & (her2==0))

print(f"\n  Subgroup sizes:")
for name, mask, T_vec in [
    ('TNBC',          tnbc,    T_chemo),
    ('All patients',  np.ones(len(df), dtype=bool), T_chemo),
    ('HER2-',         her2neg, T_chemo),
    ('TNBC excl',     tnbc,    T_chemo_excl),
    ('Luminal sanity',luminal, T_chemo),
]:
    m = mask.astype(bool)
    print(f"    {name:<20} n={m.sum():4d}  t={T_vec[m].sum():3d}  c={(T_vec[m]==0).sum():3d}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n  PRESPECIFIED ANALYSIS PLAN:")
print("  Primary    : TNBC × T_chemo")
print("  Sensitivity 1: All patients × T_chemo")
print("  Sensitivity 2: HER2- × T_chemo")
print("  Sensitivity 3: TNBC × T_chemo_excl (no hormone, no targeted)")
print("  Sanity check : ER+/PR+ HER2- × T_chemo")
print()

all_results = []

T_targeted = df['T_targeted'].astype(int).values

ANALYSES = [
    ('PRIMARY_TNBC_chemo',   tnbc,    T_chemo,      'Primary:   TNBC            × T_chemo'),
    ('SENS1_all_chemo',      np.ones(len(df), dtype=bool), T_chemo, 'Sens-1:    All patients    × T_chemo'),
    ('SENS2_HER2neg_chemo',  her2neg, T_chemo,      'Sens-2:    HER2-           × T_chemo'),
    ('SENS3_TNBC_chemo_excl',tnbc,    T_chemo_excl, 'Sens-3:    TNBC            × T_chemo_excl'),
    ('CROSSSUBGROUP_luminal',  luminal, T_chemo,      'Cross-subgroup: ER+/PR+ HER2-  × T_chemo'),
    ('FALSIF_TNBC_targeted',  tnbc,    T_targeted,   'Falsification: TNBC × T_targeted (anti-HER2 in HER2-)'),
]

def ps_diag(Xs, Ts):
    if Ts.sum() < 5 or (Ts==0).sum() < 5:
        return
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(Xs, Ts)
    ps = lr.predict_proba(Xs)[:, 1]
    p1 = Ts.mean()
    sw = np.where(Ts==1, p1/ps.clip(0.05,0.95), (1-p1)/(1-ps.clip(0.05,0.95)))
    sw = sw / sw.mean()
    ess = int((sw.sum()**2) / (sw**2).sum())
    smd = []
    for i in range(min(Xs.shape[1], 10)):
        x = Xs[:,i]; t,c = x[Ts==1], x[Ts==0]
        std = np.sqrt((t.std()**2+c.std()**2)/2)+1e-8
        smd.append(abs((t.mean()-c.mean())/std))
    print(f"    PS: [{ps.min():.3f},{ps.max():.3f}]  maxIPW={sw.max():.1f}  ESS={ess}/{len(Ts)}  SMD={np.mean(smd):.3f}")

for label, mask, T_vec, desc in ANALYSES:
    print(f"\n  ── {desc} ──")
    mask = mask.astype(bool)
    Xs   = X_all[mask]
    Ts   = T_vec[mask].astype(int)
    Ys   = Y_all[mask]
    n1   = int(Ts.sum()); n0 = int((Ts==0).sum())
    print(f"    n={mask.sum()}, treated={n1}, control={n0}")
    ps_diag(Xs, Ts)

    res = run_repeated(Xs, Ts, Ys, label)
    for r in res: r['variant'] = 'baseline'; all_results.append(r)

    if n1 < MIN_N or n0 < MIN_N:
        continue

    print(f"    [+ stabilized IPW]")
    sw = stabilized_ipw(Xs, Ts)
    res_ipw = run_repeated(Xs, Ts, Ys, label + '_sipw', sample_weight=sw)
    for r in res_ipw: r['variant'] = 'stabilized_ipw'; all_results.append(r)

    lr_ps = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr_ps.fit(Xs, Ts)
    ps   = lr_ps.predict_proba(Xs)[:, 1]
    keep = (ps >= 0.10) & (ps <= 0.90)
    print(f"    [+ PS trim 0.10–0.90]  kept={keep.sum()}/{len(keep)}")
    if keep.sum() > 40:
        res_trim = run_repeated(Xs[keep], Ts[keep], Ys[keep], label + '_trim')
        for r in res_trim: r['variant'] = 'ps_trim_10_90'; all_results.append(r)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)

df_res = pd.DataFrame([{k:v for k,v in r.items() if k!='all_ates'} for r in all_results])
df_res.to_csv(OUTPUT_DIR / 'chemo_final_results.csv', index=False)

primary = df_res[df_res['label'].str.startswith('PRIMARY')]
sanity  = df_res[df_res['label'].str.startswith('FALSIF')]
sens    = df_res[df_res['label'].str.startswith('SENS')]

print("\n  PRIMARY (TNBC × T_chemo):")
print(primary[['model','variant','n_treated','n_control',
               'median_ATE','IQR_lo','IQR_hi','ATE_min','ATE_max',
               'pct_protective','stable','direction']].to_string(index=False))

print("\n  SENSITIVITY analyses:")
print(sens[['label','model','variant','median_ATE','pct_protective','stable']].to_string(index=False))

print("\n  FALSIFICATION (TNBC × T_targeted — anti-HER2 in HER2-, should NOT be protective):")
print(sanity[['model','variant','median_ATE','pct_protective','stable']].to_string(index=False))

if len(primary) > 0:
    prot = (primary['pct_protective'] >= 60).mean()
    print(f"\n  Primary: {prot*100:.0f}% of model/variant combos show protective direction")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, mname in zip(axes, ['LinearDML','CausalForestDML','LinearDRLearner']):
    sub = df_res[df_res['model']==mname].copy()
    sub['y_label'] = sub['label'].str.replace('_chemo','').str.replace('_sipw','_ipw') + '|' + sub['variant']
    colors = ['#2ca02c' if v < -0.003 else '#ff7f0e' if abs(v) <= 0.003 else '#d62728'
              for v in sub['median_ATE']]
    ax.barh(range(len(sub)), sub['median_ATE'], color=colors, alpha=0.8)
    ax.errorbar(sub['median_ATE'], range(len(sub)),
                xerr=[sub['median_ATE']-sub['IQR_lo'], sub['IQR_hi']-sub['median_ATE']],
                fmt='none', color='black', capsize=3, lw=1.2)
    ax.axvline(0, color='gray', linestyle='--', lw=1)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub['y_label'], fontsize=7)
    ax.set_xlabel('Median ATE (10 seeds, IQR bars)')
    ax.set_title(mname, fontweight='bold', fontsize=9)

plt.suptitle('Chemotherapy — Prespecified Analysis\n(green=protective, orange=near-zero, red=harmful)', fontsize=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'chemo_final_figure.png', bbox_inches='tight', dpi=150)
plt.close(fig)

print(f"\n  Saved: {OUTPUT_DIR}")
print("=" * 65)
print("  DONE")
print("=" * 65)