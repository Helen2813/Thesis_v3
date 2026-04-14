"""
Targeted Therapy — Analysis with Small-Sample Techniques
==========================================================
Clinical question: Does adding targeted therapy (trastuzumab) ON TOP OF
chemotherapy improve survival in HER2+ patients?

Control group fix: HER2+ patients who received chemo WITHOUT targeted
(not "any untreated HER2+ patient" as before).

Small-sample techniques evaluated:
  T1. Baseline DML/CF/DR (3-fold CV)
  T2. Stabilized IPW
  T3. PS trimming
  T4. Ridge-penalized outcome model (reduces overfitting on small N)
  T5. Logistic outcome model (better calibrated for binary Y, small N)
  T6. AIPW (Augmented IPW) — manual doubly robust, good for small N
  T7. Simple IPW-weighted outcome difference (non-parametric baseline)

Falsification: HER2- chemo patients × T_targeted (should be near-zero)
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
from sklearn.linear_model import LogisticRegression, RidgeCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
SEEDS        = [42, 123, 456, 789, 1337, 2024, 31, 99, 7, 404]
CV_FOLDS     = 3
MIN_N        = 8

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR   = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]
     if (p / '02_ITE' / '01_preprocessing' / 'output' / 'ite_ready_dataset_v2.csv').exists()),
    None
)
if BASE_DIR is None:
    print("ERROR: ite_ready_dataset_v2.csv not found")
    sys.exit(1)

INPUT_DIR  = BASE_DIR / '02_ITE' / '01_preprocessing' / 'output'
OUTPUT_DIR = SCRIPT_DIR / 'targeted_final_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  TARGETED THERAPY — SMALL SAMPLE ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_DIR / 'ite_ready_dataset_v2.csv')

drop_cols = ['T','Y','patient_id','propensity_score',
             'T_hormone','T_chemo','T_targeted','T_radiation','T_hormone_excl']
X_df  = df.drop(columns=drop_cols, errors='ignore')
X_all = X_df.values.astype(float)
FEAT  = X_df.columns.tolist()
Y_all = df['Y'].astype(int).values

er   = (df['ER_status'].fillna(0) > 0.5).astype(int).values
pr   = (df['PR_status'].fillna(0) > 0.5).astype(int).values
her2 = (df['HER2_status'].fillna(0) > 0.5).astype(int).values

T_targeted = df['T_targeted'].astype(int).values
T_chemo    = df['T_chemo'].astype(int).values

# ── Correct control group definition ─────────────────────────────────────────
# Primary: HER2+ patients who got chemo, compare targeted+chemo vs chemo only
her2pos     = (her2 == 1)
her2pos_chemo = her2pos & (T_chemo == 1)   # HER2+ who received chemo (treated+control)
her2neg_chemo = (her2 == 0) & (T_chemo == 1)  # HER2- chemo patients (falsification)

print(f"\n  CONTROL GROUP: HER2+ chemo patients only")
print(f"  Clinical question: chemo+targeted vs chemo alone in HER2+")
print()

for name, mask, T_vec in [
    ('HER2+ chemo patients (primary)',  her2pos_chemo, T_targeted),
    ('All HER2+ (old approach)',         her2pos,       T_targeted),
    ('HER2- chemo patients (falsif)',   her2neg_chemo, T_targeted),
    ('All patients',                     np.ones(len(df),dtype=bool), T_targeted),
]:
    m  = mask.astype(bool)
    t  = T_vec[m].sum(); c = (T_vec[m]==0).sum()
    print(f"    {name:<40} n={m.sum():4d}  t={t:3d}  c={c:3d}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def rf_reg(seed=RANDOM_STATE):
    return RandomForestRegressor(n_estimators=200, max_depth=4,
                                 min_samples_leaf=5, max_features='sqrt',
                                 n_jobs=-1, random_state=seed)

def rf_clf(seed=RANDOM_STATE):
    return RandomForestClassifier(n_estimators=200, max_depth=4,
                                  min_samples_leaf=5, max_features='sqrt',
                                  class_weight='balanced', n_jobs=-1, random_state=seed)

def logistic_outcome(seed=RANDOM_STATE):
    return LogisticRegressionCV(cv=3, max_iter=500, random_state=seed,
                                class_weight='balanced', solver='lbfgs')

def ridge_outcome(seed=RANDOM_STATE):
    return RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

def ps_diag(Xs, Ts):
    if Ts.sum() < 3 or (Ts==0).sum() < 3:
        return
    lr  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(Xs, Ts)
    ps  = lr.predict_proba(Xs)[:, 1]
    p1  = Ts.mean()
    sw  = np.where(Ts==1, p1/ps.clip(0.05,0.95), (1-p1)/(1-ps.clip(0.05,0.95)))
    sw  = sw / sw.mean()
    ess = int((sw.sum()**2) / (sw**2).sum())
    smd = []
    for i in range(min(Xs.shape[1], 10)):
        x = Xs[:,i]; t, c = x[Ts==1], x[Ts==0]
        std = np.sqrt((t.std()**2+c.std()**2)/2)+1e-8
        smd.append(abs((t.mean()-c.mean())/std))
    print(f"    PS: [{ps.min():.3f},{ps.max():.3f}]  maxIPW={sw.max():.1f}  "
          f"ESS={ess}/{len(Ts)}  SMD={np.mean(smd):.3f}")

def stabilized_ipw(X, T, clip=(0.05, 0.95)):
    lr  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X, T)
    ps  = lr.predict_proba(X)[:, 1].clip(*clip)
    p1  = T.mean()
    sw  = np.where(T==1, p1/ps, (1-p1)/(1-ps))
    return sw / sw.mean()


# ── T7: Simple IPW estimator (non-parametric, good for tiny N) ────────────────
def simple_ipw_ate(Xs, Ts, Ys, clip=(0.05, 0.95)):
    lr  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(Xs, Ts)
    ps  = lr.predict_proba(Xs)[:, 1].clip(*clip)
    ate = float((Ts * Ys / ps - (1-Ts) * Ys / (1-ps)).mean())
    return ate

# ── T6: AIPW (Augmented IPW) ──────────────────────────────────────────────────
def aipw_ate(Xs, Ts, Ys, n_folds=3):
    """Manual AIPW — doubly robust, works well on small N."""
    n   = len(Ys)
    mu1 = np.zeros(n); mu0 = np.zeros(n); ps = np.zeros(n)
    kf  = StratifiedKFold(n_splits=min(n_folds, int(Ts.sum()), int((Ts==0).sum())),
                          shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, val_idx in kf.split(Xs, Ts):
        Xtr, Xval = Xs[tr_idx], Xs[val_idx]
        Ttr, Ytr  = Ts[tr_idx], Ys[tr_idx]
        # Outcome model
        lr_y = LogisticRegression(max_iter=500, random_state=RANDOM_STATE,
                                  class_weight='balanced')
        Xtr1 = Xtr[Ttr==1]; Xtr0 = Xtr[Ttr==0]
        Ytr1 = Ytr[Ttr==1]; Ytr0 = Ytr[Ttr==0]
        if len(Xtr1) > 1 and len(np.unique(Ytr1)) > 1:
            lr_y.fit(Xtr1, Ytr1)
            mu1[val_idx] = lr_y.predict_proba(Xval)[:, 1]
        else:
            mu1[val_idx] = Ytr1.mean() if len(Ytr1) > 0 else 0.1
        if len(Xtr0) > 1 and len(np.unique(Ytr0)) > 1:
            lr_y.fit(Xtr0, Ytr0)
            mu0[val_idx] = lr_y.predict_proba(Xval)[:, 1]
        else:
            mu0[val_idx] = Ytr0.mean() if len(Ytr0) > 0 else 0.1
        # Propensity model
        lr_t = LogisticRegression(max_iter=500, random_state=RANDOM_STATE,
                                  class_weight='balanced')
        lr_t.fit(Xtr, Ttr)
        ps[val_idx] = lr_t.predict_proba(Xval)[:, 1].clip(0.05, 0.95)

    aipw = (mu1 - mu0
            + Ts * (Ys - mu1) / ps
            - (1 - Ts) * (Ys - mu0) / (1 - ps))
    return float(aipw.mean())


# ── DML repeated seeds ────────────────────────────────────────────────────────
def run_dml_repeated(Xs, Ts, Ys, label, model_y_fn, model_t_fn, tag, sw=None):
    n1 = int(Ts.sum()); n0 = int((Ts==0).sum())
    if n1 < MIN_N or n0 < MIN_N:
        return []
    lin_ates, cf_ates, dr_ates = [], [], []
    for seed in SEEDS:
        try:
            cv = min(CV_FOLDS, n1, n0)
            lin = LinearDML(model_y=model_y_fn(seed), model_t=model_t_fn(seed),
                            discrete_treatment=True, linear_first_stages=False,
                            cv=cv, random_state=seed)
            kw = {'inference': 'statsmodels'}
            if sw is not None: kw['sample_weight'] = sw
            lin.fit(Ys.astype(float), Ts, X=Xs.astype(float), **kw)
            lin_ates.append(float(lin.effect(Xs.astype(float)).mean()))

            cf = CausalForestDML(model_y=model_y_fn(seed), model_t=model_t_fn(seed),
                                 discrete_treatment=True, n_estimators=200,
                                 min_samples_leaf=5, max_features='sqrt',
                                 cv=cv, random_state=seed, n_jobs=-1)
            cf_kw = {} if sw is None else {'sample_weight': sw}
            cf.fit(Ys.astype(float), Ts, X=Xs.astype(float), **cf_kw)
            cf_ates.append(float(cf.effect(Xs.astype(float)).mean()))

            dr = LinearDRLearner(model_regression=model_y_fn(seed),
                                 model_propensity=model_t_fn(seed),
                                 cv=cv, random_state=seed)
            dr_kw = {} if sw is None else {'sample_weight': sw}
            dr.fit(Ys.astype(float), Ts, X=Xs.astype(float), **dr_kw)
            dr_ates.append(float(dr.effect(Xs.astype(float)).mean()))
        except Exception as e:
            pass

    results = []
    for ates, mname in [(lin_ates,'LinearDML'),(cf_ates,'CausalForestDML'),(dr_ates,'LinearDRLearner')]:
        if not ates: continue
        a  = np.array(ates)
        med = float(np.median(a))
        prot = float((a < 0).mean() * 100)
        stable = prot >= 80 or prot <= 20
        print(f"    [{tag}] {mname:<20} median={med:+.4f}  "
              f"range=[{a.min():+.4f},{a.max():+.4f}]  "
              f"protective={prot:.0f}%  {'✓' if stable else '~'}")
        results.append(dict(label=label, technique=tag, model=mname,
                            n_treated=n1, n_control=n0,
                            median_ATE=med, ATE_min=float(a.min()),
                            ATE_max=float(a.max()), IQR_lo=float(np.percentile(a,25)),
                            IQR_hi=float(np.percentile(a,75)),
                            pct_protective=prot, stable=stable))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY: HER2+ chemo patients, compare targeted vs no targeted
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PRIMARY: HER2+ chemo patients × T_targeted")
print("  (chemo+targeted vs chemo only — correct clinical comparison)")
print("=" * 70)

mask_primary = her2pos_chemo.astype(bool)
Xs = X_all[mask_primary]
Ts = T_targeted[mask_primary].astype(int)
Ys = Y_all[mask_primary]
n1 = int(Ts.sum()); n0 = int((Ts==0).sum())
print(f"\n  n={mask_primary.sum()}, treated={n1}, control={n0}")
ps_diag(Xs, Ts)

all_results = []

# T1 — RF baseline
print("\n  T1. RF baseline (DML/CF/DR)")
res = run_dml_repeated(Xs, Ts, Ys, 'PRIMARY', rf_reg, rf_clf, 'T1_rf_baseline')
all_results.extend(res)

# T2 — Stabilized IPW
print("\n  T2. + Stabilized IPW")
sw = stabilized_ipw(Xs, Ts)
res = run_dml_repeated(Xs, Ts, Ys, 'PRIMARY', rf_reg, rf_clf, 'T2_sipw', sw=sw)
all_results.extend(res)

# T3 — PS trimming
lr_ps = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
lr_ps.fit(Xs, Ts)
ps_vals = lr_ps.predict_proba(Xs)[:, 1]
keep = (ps_vals >= 0.10) & (ps_vals <= 0.90)
print(f"\n  T3. PS trimming [0.10,0.90]  kept={keep.sum()}/{len(keep)}")
if keep.sum() > 15 and Ts[keep].sum() >= MIN_N:
    res = run_dml_repeated(Xs[keep], Ts[keep], Ys[keep], 'PRIMARY', rf_reg, rf_clf, 'T3_ps_trim')
    all_results.extend(res)

# T4 — Ridge outcome model (penalized, better for small N)
print("\n  T4. Ridge penalized outcome model")
res = run_dml_repeated(Xs, Ts, Ys, 'PRIMARY', ridge_outcome, rf_clf, 'T4_ridge_outcome')
all_results.extend(res)

# T5 — Logistic outcome model (better calibrated for binary Y)
print("\n  T5. Logistic outcome model")
res = run_dml_repeated(Xs, Ts, Ys, 'PRIMARY', logistic_outcome, rf_clf, 'T5_logistic_outcome')
all_results.extend(res)

# T6 — AIPW (manual, cross-fitted, non-parametric)
if n1 >= MIN_N and n0 >= MIN_N:
    aipw_ates = []
    for seed in SEEDS:
        np.random.seed(seed)
        try:
            aipw_ates.append(aipw_ate(Xs, Ts, Ys))
        except:
            pass
    if aipw_ates:
        a = np.array(aipw_ates)
        prot = float((a < 0).mean() * 100)
        print(f"\n  T6. AIPW (augmented IPW, doubly robust)")
        print(f"    median={np.median(a):+.4f}  range=[{a.min():+.4f},{a.max():+.4f}]  "
              f"protective={prot:.0f}%")
        all_results.append(dict(label='PRIMARY', technique='T6_aipw', model='AIPW',
                                n_treated=n1, n_control=n0,
                                median_ATE=float(np.median(a)),
                                ATE_min=float(a.min()), ATE_max=float(a.max()),
                                IQR_lo=float(np.percentile(a,25)),
                                IQR_hi=float(np.percentile(a,75)),
                                pct_protective=prot, stable=(prot>=80 or prot<=20)))

# T7 — Simple IPW (most transparent for small N)
if n1 >= MIN_N and n0 >= MIN_N:
    simple_ates = []
    for seed in SEEDS:
        np.random.seed(seed)
        try:
            simple_ates.append(simple_ipw_ate(Xs, Ts, Ys))
        except:
            pass
    if simple_ates:
        a = np.array(simple_ates)
        prot = float((a < 0).mean() * 100)
        print(f"\n  T7. Simple IPW (non-parametric, most transparent)")
        print(f"    median={np.median(a):+.4f}  range=[{a.min():+.4f},{a.max():+.4f}]  "
              f"protective={prot:.0f}%")
        all_results.append(dict(label='PRIMARY', technique='T7_simple_ipw', model='SimpleIPW',
                                n_treated=n1, n_control=n0,
                                median_ATE=float(np.median(a)),
                                ATE_min=float(a.min()), ATE_max=float(a.max()),
                                IQR_lo=float(np.percentile(a,25)),
                                IQR_hi=float(np.percentile(a,75)),
                                pct_protective=prot, stable=(prot>=80 or prot<=20)))


# ═══════════════════════════════════════════════════════════════════════════════
# FALSIFICATION: HER2- chemo patients × T_targeted
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FALSIFICATION: HER2- chemo patients × T_targeted")
print("  (anti-HER2 in HER2-negative — should be near-zero/harmful)")
print("=" * 70)

mask_f = her2neg_chemo.astype(bool)
Xf = X_all[mask_f]; Tf = T_targeted[mask_f].astype(int); Yf = Y_all[mask_f]
nf1 = int(Tf.sum()); nf0 = int((Tf==0).sum())
print(f"\n  n={mask_f.sum()}, treated={nf1}, control={nf0}")
ps_diag(Xf, Tf)

falsif_res = run_dml_repeated(Xf, Tf, Yf, 'FALSIF', rf_reg, rf_clf, 'T1_baseline')
all_results.extend(falsif_res)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  RESULTS SUMMARY  (⚠ EXPLORATORY — n_treated≈26)")
print("=" * 70)

df_res = pd.DataFrame(all_results)
df_res.to_csv(OUTPUT_DIR / 'targeted_final_results.csv', index=False)

primary = df_res[df_res['label']=='PRIMARY']
falsif  = df_res[df_res['label']=='FALSIF']

print("\n  PRIMARY — technique comparison:")
print(primary[['technique','model','median_ATE','ATE_min','ATE_max',
               'pct_protective','stable']].to_string(index=False))

if len(falsif) > 0:
    print("\n  FALSIFICATION (HER2- chemo × T_targeted):")
    print(falsif[['technique','model','median_ATE','pct_protective','stable']].to_string(index=False))

# Direction agreement across techniques
prot_pct = (primary['pct_protective'] >= 60).mean()
print(f"\n  Technique agreement: {prot_pct*100:.0f}% show protective direction")
print(f"  Number of techniques/models evaluated: {len(primary)}")

# Sign stability summary
signs = primary['median_ATE'].apply(lambda x: 'protective' if x < 0 else 'harmful')
print(f"  Sign distribution: {signs.value_counts().to_dict()}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
sub = primary.copy()
sub['y_label'] = sub['technique'] + ' | ' + sub['model']
colors = ['#2ca02c' if v < -0.003 else '#ff7f0e' if abs(v)<=0.003 else '#d62728'
          for v in sub['median_ATE']]
ax.barh(range(len(sub)), sub['median_ATE'], color=colors, alpha=0.8)
ax.errorbar(sub['median_ATE'], range(len(sub)),
            xerr=[sub['median_ATE']-sub['IQR_lo'], sub['IQR_hi']-sub['median_ATE']],
            fmt='none', color='black', capsize=3, lw=1.2)
ax.axvline(0, color='gray', linestyle='--', lw=1)
ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub['y_label'], fontsize=7)
ax.set_xlabel('Median ATE (10 seeds)')
ax.set_title('PRIMARY: HER2+ chemo × T_targeted\n(chemo+targeted vs chemo alone)',
             fontweight='bold', fontsize=9)

ax2 = axes[1]
if len(falsif) > 0:
    colors_f = ['#2ca02c' if v < -0.003 else '#ff7f0e' if abs(v)<=0.003 else '#d62728'
                for v in falsif['median_ATE']]
    ax2.barh(range(len(falsif)), falsif['median_ATE'], color=colors_f, alpha=0.8)
    ax2.axvline(0, color='gray', linestyle='--', lw=1)
    ax2.set_yticks(range(len(falsif)))
    ax2.set_yticklabels(falsif['technique'] + '|' + falsif['model'], fontsize=8)
    ax2.set_xlabel('Median ATE')
    ax2.set_title('FALSIFICATION: HER2- chemo × T_targeted\n(should be near-zero)',
                  fontweight='bold', fontsize=9)
else:
    ax2.text(0.5, 0.5, 'No falsification cases\n(n_treated=0)',
             ha='center', va='center', fontsize=10)
    ax2.axis('off')

plt.suptitle('Targeted Therapy — Small Sample Analysis\n⚠ Exploratory: n_treated≈26',
             fontsize=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'targeted_final_figure.png', bbox_inches='tight', dpi=150)
plt.close(fig)

print(f"\n  Saved: {OUTPUT_DIR}")
print("=" * 70)
print("  DONE")
print("=" * 70)