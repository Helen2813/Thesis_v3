"""
Ablation Study: Systematic Hyperparameter Search for ITE Estimation
====================================================================
Thesis: Causal Multimodal Analysis of Breast Cancer Survival (TCGA-BRCA)

Runs N experiments, each with different config.
Saves full comparison table + best config automatically.

Run:     python ablation_study.py
Install: pip install econml xgboost imbalanced-learn tqdm
"""

import os, sys, json, warnings, logging, itertools
from pathlib import Path
from time import time
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

try:
    from econml.dml import LinearDML, CausalForestDML
except ImportError:
    log.error("pip install econml"); sys.exit(1)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("XGBoost not found — some experiments will use RF")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    log.warning("imbalanced-learn not found — SMOTE experiments skipped. pip install imbalanced-learn")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_BOOT       = 1000   # faster for ablation
PRIMARY_ARM  = 'T_any'  # evaluate all experiments on this arm first

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR   = SCRIPT_DIR.parent.parent
INPUT_DIR  = BASE_DIR / '02_ITE' / '01_preprocessing' / 'output'
OUTPUT_DIR = SCRIPT_DIR / 'ablation_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
ite_df = pd.read_csv(INPUT_DIR / 'ite_ready_dataset.csv')
with open(INPUT_DIR / 'preprocessing_metadata.json') as f:
    meta = json.load(f)

CLASS_WEIGHT = {int(k): float(v)
                for k,v in meta.get('class_weight_balanced',
                                    {'0':0.55,'1':5.38}).items()}

X_df  = ite_df.drop(columns=['T','Y','patient_id','propensity_score'], errors='ignore')
FEAT  = X_df.columns.tolist()
X     = X_df.values.astype(float)
Y     = ite_df['Y'].astype(int).values
T_bin = ite_df['T'].astype(int).values

# Build subgroup masks
has_receptor = all(c in X_df.columns for c in ['ER_status','PR_status','HER2_status'])
if has_receptor:
    er   = (X_df['ER_status'].fillna(0).values   > 0.5).astype(int)
    pr   = (X_df['PR_status'].fillna(0).values   > 0.5).astype(int)
    her2 = (X_df['HER2_status'].fillna(0).values > 0.5).astype(int)
    SUBGROUP_MASKS = {
        'T_any':      np.ones(len(T_bin), dtype=bool),
        'T_hormone':  ((er|pr) & (her2==0)).astype(bool),
        'T_targeted': (her2==1).astype(bool),
        'T_chemo':    ((er==0)&(pr==0)&(her2==0)).astype(bool),
    }
    T_MATRIX = {
        'T_any':      T_bin,
        'T_hormone':  T_bin,
        'T_targeted': T_bin,
        'T_chemo':    T_bin,
    }
    T_COLS = ['T_any','T_hormone','T_targeted','T_chemo']
else:
    SUBGROUP_MASKS = {'T_any': np.ones(len(T_bin), dtype=bool)}
    T_MATRIX = {'T_any': T_bin}
    T_COLS = ['T_any']

TREATMENT_LABELS = {
    'T_any':      'Any Treatment',
    'T_hormone':  'Hormone Therapy',
    'T_targeted': 'Targeted Therapy',
    'T_chemo':    'Chemotherapy',
}

print(f"Loaded: {X.shape[0]} patients, {X.shape[1]} features, "
      f"Y=1:{Y.sum()} ({Y.mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # ── Baseline ──────────────────────────────────────────────────────────────
    {
        'name':        'EXP01_baseline_RF',
        'description': 'Baseline: Random Forest, no trimming, 5-fold, no reweighting',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     None,
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    # ── XGBoost base ──────────────────────────────────────────────────────────
    {
        'name':        'EXP02_XGBoost_base',
        'description': 'XGBoost base estimator (better for imbalanced small samples)',
        'model_y':     'xgb',
        'model_t':     'xgb',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     None,
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    # ── PS trimming ───────────────────────────────────────────────────────────
    {
        'name':        'EXP03_PS_trim_05_95',
        'description': 'Propensity trimming [0.05, 0.95] — remove poor overlap',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     (0.05, 0.95),
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    {
        'name':        'EXP04_PS_trim_10_90',
        'description': 'Propensity trimming [0.10, 0.90] — stricter overlap',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     (0.10, 0.90),
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    # ── Sample reweighting ────────────────────────────────────────────────────
    {
        'name':        'EXP05_reweight_balanced',
        'description': 'Balanced sample weights for Y imbalance (9.3% events)',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     None,
        'reweight':    True,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    # ── CV folds ──────────────────────────────────────────────────────────────
    {
        'name':        'EXP06_3fold_CV',
        'description': '3-fold CV (better for small subgroups n<300)',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          3,
        'ps_trim':     None,
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    # ── CF hyperparameters ────────────────────────────────────────────────────
    {
        'name':        'EXP07_CF_500est',
        'description': 'CausalForest n_estimators=500 (more stable estimates)',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    500,
        'cv':          5,
        'ps_trim':     None,
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    {
        'name':        'EXP08_CF_min_leaf_5',
        'description': 'CausalForest min_samples_leaf=5 (more heterogeneity)',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     None,
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 5,
    },
    {
        'name':        'EXP09_CF_min_leaf_20',
        'description': 'CausalForest min_samples_leaf=20 (smoother, less overfit)',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     None,
        'reweight':    False,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 20,
    },
    # ── SMOTE ─────────────────────────────────────────────────────────────────
    {
        'name':        'EXP10_SMOTE',
        'description': 'SMOTE oversampling Y=1 to reduce 1:10 imbalance',
        'model_y':     'rf',
        'model_t':     'rf',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     None,
        'reweight':    False,
        'smote':       True,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    # ── Best combo ────────────────────────────────────────────────────────────
    {
        'name':        'EXP11_best_combo_A',
        'description': 'Best combo A: XGBoost + PS trim[0.05,0.95] + reweight + 3CV',
        'model_y':     'xgb',
        'model_t':     'xgb',
        'n_est_cf':    500,
        'cv':          3,
        'ps_trim':     (0.05, 0.95),
        'reweight':    True,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 5,
    },
    {
        'name':        'EXP12_best_combo_B',
        'description': 'Best combo B: XGBoost + PS trim[0.10,0.90] + reweight + 5CV',
        'model_y':     'xgb',
        'model_t':     'xgb',
        'n_est_cf':    500,
        'cv':          5,
        'ps_trim':     (0.10, 0.90),
        'reweight':    True,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   6,
        'cf_min_leaf': 10,
    },
    {
        'name':        'EXP13_GBM_base',
        'description': 'GradientBoosting sklearn (alternative to XGBoost)',
        'model_y':     'gbm',
        'model_t':     'gbm',
        'n_est_cf':    300,
        'cv':          5,
        'ps_trim':     (0.05, 0.95),
        'reweight':    True,
        'smote':       False,
        'min_leaf':    10,
        'max_depth':   4,
        'cf_min_leaf': 10,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def make_model(kind, depth=6, n_est=300, seed=RANDOM_STATE, is_outcome=True):
    if kind == 'xgb' and HAS_XGB:
        return XGBRegressor(
            n_estimators=n_est, max_depth=depth,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1,
            random_state=seed, verbosity=0
        )
    elif kind == 'gbm':
        return GradientBoostingRegressor(
            n_estimators=n_est, max_depth=depth,
            learning_rate=0.05, subsample=0.8,
            random_state=seed
        )
    else:  # rf
        return RandomForestRegressor(
            n_estimators=n_est, max_depth=depth,
            min_samples_leaf=10, max_features='sqrt',
            n_jobs=-1, random_state=seed
        )

def get_trim_mask(X_sub, T_sub, bounds):
    if bounds is None:
        return np.ones(len(T_sub), dtype=bool)
    lr  = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    lr.fit(X_sub, T_sub)
    ps  = lr.predict_proba(X_sub)[:,1]
    return (ps >= bounds[0]) & (ps <= bounds[1])

def apply_smote(X_sub, T_sub, Y_sub, seed=RANDOM_STATE):
    if not HAS_SMOTE:
        return X_sub, T_sub, Y_sub
    try:
        # SMOTE on outcome (Y), keeping T alignment
        XY = np.column_stack([X_sub, T_sub])
        sm = SMOTE(random_state=seed, k_neighbors=min(5, (Y_sub==1).sum()-1))
        XY_res, Y_res = sm.fit_resample(XY, Y_sub)
        return XY_res[:,:-1], XY_res[:,-1].astype(int), Y_res
    except Exception as ex:
        log.warning(f"SMOTE failed: {ex}")
        return X_sub, T_sub, Y_sub

def compute_metrics(ite, T_arm, Y_arm, n_bins=100, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    # ATE bootstrap CI
    boots = [rng.choice(ite, len(ite), replace=True).mean() for _ in range(n_boot)]
    ate   = float(ite.mean())
    ate_lo, ate_hi = float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))

    # ATT
    ite1 = ite[T_arm==1]
    b1   = [rng.choice(ite1, len(ite1), replace=True).mean() for _ in range(n_boot)]
    att  = float(ite1.mean()) if len(ite1) > 0 else np.nan
    att_lo = float(np.percentile(b1,2.5)) if len(ite1)>0 else np.nan
    att_hi = float(np.percentile(b1,97.5)) if len(ite1)>0 else np.nan

    # AUUC / Qini
    order = np.argsort(-ite)
    T_o, Y_o = T_arm[order], Y_arm[order]
    n = len(T_o)
    fracs, uplifts = [], []
    for ki in range(1, n_bins+1):
        tk = int(n*ki/n_bins)
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
    raw     = Y_arm[T_arm==1].mean() - Y_arm[T_arm==0].mean()
    qini    = auuc - raw * 0.5

    # Policy value (IPW)
    lr2 = LogisticRegression(max_iter=500, random_state=seed)
    X_dummy = np.column_stack([ite, np.ones(len(ite))])
    lr2.fit(X_dummy, T_arm)
    ps2  = lr2.predict_proba(X_dummy)[:,1].clip(0.05,0.95)
    ipw1 = np.where(T_arm==1, Y_arm/ps2, 0)
    ipw0 = np.where(T_arm==0, Y_arm/(1-ps2), 0)
    pol  = (ite < 0).astype(int)
    policy_gain = float(Y_arm[T_arm==0].mean() - (pol*ipw1+(1-pol)*ipw0).mean())

    return {
        'ATE': ate, 'ATE_lo': ate_lo, 'ATE_hi': ate_hi,
        'ATT': att, 'ATT_lo': att_lo, 'ATT_hi': att_hi,
        'AUUC': auuc, 'Qini': qini,
        'policy_gain': policy_gain,
        'pct_benefit': float((ite<0).mean()*100),
        'ITE_std': float(ite.std()),
        'fracs': fracs, 'uplifts': uplifts,
    }

# ─────────────────────────────────────────────────────────────────────────────
# RUN EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning {len(EXPERIMENTS)} experiments × {len(T_COLS)} arms "
      f"× 2 models = {len(EXPERIMENTS)*len(T_COLS)*2} fits\n")

all_results = []  # list of dicts for comparison table

for exp_idx, cfg in enumerate(EXPERIMENTS):
    name = cfg['name']
    desc = cfg['description']

    if cfg['smote'] and not HAS_SMOTE:
        log.warning(f"Skipping {name}: SMOTE not installed")
        continue
    if cfg['model_y'] == 'xgb' and not HAS_XGB:
        log.warning(f"Skipping {name}: XGBoost not installed")
        continue

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  {desc}")
    print(f"{'='*65}")

    exp_row_base = {
        'experiment': name,
        'description': desc,
        'model_y': cfg['model_y'],
        'cv': cfg['cv'],
        'ps_trim': str(cfg['ps_trim']),
        'reweight': cfg['reweight'],
        'smote': cfg['smote'],
        'cf_min_leaf': cfg['cf_min_leaf'],
        'n_est_cf': cfg['n_est_cf'],
    }

    for tc in T_COLS:
        mask_base = SUBGROUP_MASKS[tc]

        # Propensity trimming
        Xs_full = X[mask_base]
        Ts_full = T_MATRIX[tc][mask_base].astype(float)
        Ys_full = Y[mask_base].astype(float)

        trim_keep = get_trim_mask(Xs_full, Ts_full, cfg['ps_trim'])
        Xs = Xs_full[trim_keep]
        Ts = Ts_full[trim_keep]
        Ys = Ys_full[trim_keep]
        n1, n0 = int(Ts.sum()), int((Ts==0).sum())

        if n1 < 10 or n0 < 10:
            continue

        # SMOTE
        if cfg['smote']:
            Xs, Ts, Ys = apply_smote(Xs, Ts.astype(int), Ys.astype(int))
            Ts = Ts.astype(float); Ys = Ys.astype(float)

        # Sample weights
        sw = compute_sample_weight('balanced', Ys) if cfg['reweight'] else None

        cv_k = min(cfg['cv'], n1, n0)

        my = make_model(cfg['model_y'], depth=cfg['max_depth'], seed=RANDOM_STATE)
        mt = make_model(cfg['model_t'], depth=cfg['max_depth'], seed=RANDOM_STATE)

        for model_type in ['LinearDML', 'CausalForestDML']:
            t0 = time()
            try:
                if model_type == 'LinearDML':
                    mdl = LinearDML(
                        model_y=deepcopy(my), model_t=deepcopy(mt),
                        linear_first_stages=False,
                        cv=cv_k, random_state=RANDOM_STATE,
                    )
                    fit_kwargs = {'inference': 'statsmodels'}
                    if sw is not None: fit_kwargs['sample_weight'] = sw
                    mdl.fit(Ys, Ts, X=Xs, **fit_kwargs)
                else:
                    mdl = CausalForestDML(
                        model_y=deepcopy(my), model_t=deepcopy(mt),
                        n_estimators=cfg['n_est_cf'],
                        min_samples_leaf=cfg['cf_min_leaf'],
                        max_features='sqrt',
                        cv=cv_k, random_state=RANDOM_STATE, n_jobs=-1,
                    )
                    fit_kwargs = {}
                    if sw is not None: fit_kwargs['sample_weight'] = sw
                    mdl.fit(Ys, Ts, X=Xs, **fit_kwargs)

                ite = mdl.effect(Xs, T0=np.zeros(len(Xs)),
                                 T1=np.ones(len(Xs))).flatten()
                T_eval = Ts.astype(int)
                Y_eval = Ys.astype(int)

                m = compute_metrics(ite, T_eval, Y_eval)
                elapsed = time() - t0

                sig = '✓' if not (m['ATE_lo']<0<m['ATE_hi']) else '○'
                print(f"  {model_type:<18} | {tc:<14} | "
                      f"ATE={m['ATE']:+.4f} [{m['ATE_lo']:+.4f},{m['ATE_hi']:+.4f}]  "
                      f"AUUC={m['AUUC']:.4f}  Qini={m['Qini']:.4f}  "
                      f"PG={m['policy_gain']:+.4f}  {sig}  [{elapsed:.0f}s]")

                row = {**exp_row_base,
                       'model_type': model_type,
                       'arm': tc,
                       'arm_label': TREATMENT_LABELS[tc],
                       'n_patients': len(Xs),
                       'n_treated': n1,
                       'n_control': n0,
                       'n_trimmed': int((~trim_keep).sum()),
                       **{k: round(v,5) if isinstance(v,float) else v
                          for k,v in m.items()
                          if isinstance(v,(float,int))},
                       'ATE_sig': sig,
                       'elapsed_s': round(elapsed,1),
                }
                all_results.append(row)

            except Exception as ex:
                log.error(f"  {model_type}|{tc} FAILED: {ex}")

# ─────────────────────────────────────────────────────────────────────────────
# COMPARE & SELECT BEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ABLATION RESULTS — FULL COMPARISON")
print("="*65)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / 'ablation_all_results.csv', index=False)

# Score each experiment: higher AUUC + Qini + policy_gain = better
# Normalise per arm to make comparable across arms
score_cols = ['AUUC', 'Qini', 'policy_gain']
summary_rows = []
for exp_name in results_df['experiment'].unique():
    sub = results_df[results_df['experiment'] == exp_name]
    for model_type in sub['model_type'].unique():
        sub2 = sub[sub['model_type'] == model_type]
        # Composite score: mean of normalised AUUC + Qini + policy_gain
        auuc_mean = sub2['AUUC'].mean()
        qini_mean = sub2['Qini'].mean()
        pg_mean   = sub2['policy_gain'].mean()
        n_sig     = (sub2['ATE_sig'] == '✓').sum()
        summary_rows.append({
            'experiment':  exp_name,
            'model_type':  model_type,
            'description': sub2['description'].iloc[0],
            'mean_AUUC':   round(auuc_mean, 5),
            'mean_Qini':   round(qini_mean, 5),
            'mean_PolicyGain': round(pg_mean, 5),
            'n_significant_arms': int(n_sig),
            'total_arms':  len(sub2),
        })

summary_df = pd.DataFrame(summary_rows)

# Composite score (rank by sum of normalised metrics)
for col in ['mean_AUUC','mean_Qini','mean_PolicyGain']:
    mn, mx = summary_df[col].min(), summary_df[col].max()
    if mx > mn:
        summary_df[f'{col}_norm'] = (summary_df[col]-mn)/(mx-mn)
    else:
        summary_df[f'{col}_norm'] = 0.5

summary_df['composite_score'] = (
    summary_df['mean_AUUC_norm'] * 0.4 +
    summary_df['mean_Qini_norm'] * 0.4 +
    summary_df['mean_PolicyGain_norm'] * 0.2
).round(4)

summary_df = summary_df.sort_values('composite_score', ascending=False)
summary_df.to_csv(OUTPUT_DIR / 'ablation_summary.csv', index=False)

print("\n  EXPERIMENT RANKING (by composite score: 40% AUUC + 40% Qini + 20% PolicyGain)")
print(f"\n  {'Rank':<5} {'Experiment':<25} {'Model':<18} "
      f"{'AUUC':>8} {'Qini':>8} {'PG':>8} {'Sig':>4} {'Score':>7}")
print(f"  {'-'*5} {'-'*25} {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*4} {'-'*7}")

for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
    print(f"  {rank:<5} {row['experiment']:<25} {row['model_type']:<18} "
          f"{row['mean_AUUC']:>8.4f} {row['mean_Qini']:>8.4f} "
          f"{row['mean_PolicyGain']:>8.4f} "
          f"{row['n_significant_arms']}/{row['total_arms']:>2}  "
          f"{row['composite_score']:>7.4f}")

# Best config
best = summary_df.iloc[0]
print(f"\n  BEST CONFIG: {best['experiment']} — {best['model_type']}")
print(f"  Description: {best['description']}")
print(f"  Score: {best['composite_score']:.4f}  "
      f"AUUC={best['mean_AUUC']:.4f}  Qini={best['mean_Qini']:.4f}")

# Save best config
best_cfg = EXPERIMENTS[[e['name'] for e in EXPERIMENTS].index(best['experiment'])]
with open(OUTPUT_DIR / 'best_config.json', 'w') as f:
    json.dump({'best_experiment': best['experiment'],
               'best_model': best['model_type'],
               'description': best['description'],
               'config': best_cfg}, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE: AUUC heatmap experiments × arms
# ─────────────────────────────────────────────────────────────────────────────
for metric_col, metric_name in [('AUUC','AUUC'), ('Qini','Qini coefficient')]:
    for model_t in ['LinearDML','CausalForestDML']:
        sub = results_df[results_df['model_type']==model_t]
        if len(sub) == 0: continue
        pivot = sub.pivot_table(index='experiment', columns='arm',
                                values=metric_col, aggfunc='mean')
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2),
                                        max(4, len(pivot)*0.5)))
        import seaborn as sns
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn',
                    center=0, ax=ax, linewidths=0.5,
                    cbar_kws={'label': metric_name})
        ax.set_title(f'{model_t} — {metric_name} by Experiment × Arm\n'
                     f'(green=better, red=worse than random)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Treatment Arm')
        ax.set_ylabel('Experiment')
        plt.tight_layout()
        fname = f'heatmap_{metric_col}_{model_t}.png'
        fig.savefig(OUTPUT_DIR / fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
        log.info(f"Saved: {fname}")

# FIGURE: ATE comparison across experiments for T_any
for model_t in ['LinearDML','CausalForestDML']:
    sub = results_df[(results_df['model_type']==model_t) &
                     (results_df['arm']=='T_any')].copy()
    if len(sub) == 0: continue
    sub = sub.sort_values('composite_score' if 'composite_score' in sub.columns
                           else 'AUUC', ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(sub)*0.45)))
    for ax, (col, label) in zip(axes, [('AUUC','AUUC'), ('Qini','Qini')]):
        colors = ['#2ca02c' if v > sub[col].median() else '#d62728'
                  for v in sub[col]]
        ax.barh(sub['experiment'], sub[col], color=colors, alpha=0.8)
        ax.axvline(0, color='gray', linestyle='--', lw=1)
        ax.axvline(sub[col].median(), color='black', linestyle=':', lw=1.5,
                   label=f'median={sub[col].median():.4f}')
        ax.set_title(f'{label} — T_any arm\n{model_t}', fontweight='bold')
        ax.set_xlabel(label)
        ax.legend(fontsize=9)
    plt.suptitle(f'Experiment Comparison: {model_t}', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f'experiment_comparison_{model_t}.png',
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info(f"Saved experiment_comparison_{model_t}.png")

print(f"\n{'='*65}")
print("  ABLATION COMPLETE")
print(f"{'='*65}")
print(f"  Experiments run  : {results_df['experiment'].nunique()}")
print(f"  Total model fits : {len(results_df)}")
print(f"  Best config      : {best['experiment']} — {best['model_type']}")
print(f"  Output files in  : {OUTPUT_DIR}")
print()
print("  FILES:")
for fp in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {fp.name}")
print()
print("  NEXT STEP:")
print("  Use best_config.json in final_ite_comparison_v2.py")
print("  → replace EXPERIMENTS[0] config with best_config settings")
print(f"{'='*65}")
