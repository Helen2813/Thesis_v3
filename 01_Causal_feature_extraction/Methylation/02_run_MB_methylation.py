"""
MARKOV BLANKET - METHYLATION
============================================
Changes:
- Paths for Methylation
- Added C-index calculation
- MMMB ONLY at alpha=0.05 (too slow otherwise)
- Filters out non-dataset files
"""

import sys
import os
sys.path.insert(0, r"C:\Users\olegk\Desktop\Thesis_master2\pyCausalFS\pyCausalFS\pyCausalFS")

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from lifelines.utils import concordance_index
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from CBD.MBs.IAMB import IAMB
from CBD.MBs.GSMB import GSMB
from CBD.MBs.MMMB.MMMB import MMMB
import CBD.MBs.common.fisher_z_test as fz

# ============================================================================
# CONFIG
# ============================================================================

# PATHS FOR METHYLATION
FILTERED_DIR = r"C:\Users\olegk\Desktop\Thesis_v3\01_Causal_feature_extraction\Methylation\statistical_filtered"
OUTPUT_DIR = r"C:\Users\olegk\Desktop\Thesis_v3\01_Causal_feature_extraction\Methylation\mb_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MB_MIN_FEATURES = 50  # minimum features after MB; pad with composite fallback if fewer

# Grid search parameters
ALPHAS = [0.05, 0.10, 0.20]

# MMMB ONLY at alpha=0.05 (too slow for others!)
ALGORITHMS_BY_ALPHA = {
    0.05: ['IAMB', 'GSMB', 'MMMB'],  # MMMB only here
    0.10: ['IAMB', 'GSMB'],           # Skip MMMB - too slow
    0.20: ['IAMB', 'GSMB']            # Skip MMMB - too slow
}

print("="*80)
print("MARKOV BLANKET - METHYLATION")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nInput directory: {FILTERED_DIR}")
print(f"\nAlgorithm strategy (MMMB optimized):")
print(f"  alpha=0.05: {ALGORITHMS_BY_ALPHA[0.05]} (MMMB included)")
print(f"  alpha=0.10: {ALGORITHMS_BY_ALPHA[0.10]} (MMMB skipped - too slow)")
print(f"  alpha=0.20: {ALGORITHMS_BY_ALPHA[0.20]} (MMMB skipped - too slow)")
total_per_dataset = sum(len(algos) for algos in ALGORITHMS_BY_ALPHA.values())
print(f"\nTotal combinations per dataset: {total_per_dataset}")
print(f"\nOutput: {OUTPUT_DIR}")
print("="*80)

# ============================================================================
# FISHER-Z PATCH
# ============================================================================

def partial_corr_coef(data, x, y, z=None, ridge_lambda=1e-6):
    """Patched Fisher-Z"""
    if z is None:
        has_z = False
    elif isinstance(z, (int, np.integer)):
        has_z = True
        z = [int(z)]
    elif hasattr(z, '__len__'):
        has_z = len(z) > 0
        if has_z:
            z = [int(zi) for zi in z]
    else:
        has_z = True
        z = [int(z)]
    
    if not has_z:
        var_x, var_y = data[x, x], data[y, y]
        cov_xy = data[x, y]
        if var_x < 1e-10 or var_y < 1e-10:
            return 0.0
        r = cov_xy / np.sqrt(var_x * var_y)
        return float(np.clip(r, -0.999999, 0.999999))
    
    vars_list = [x, y] + z
    n = len(vars_list)
    sub_cov = np.zeros((n, n))
    for i, vi in enumerate(vars_list):
        for j, vj in enumerate(vars_list):
            sub_cov[i, j] = data[vi, vj]
    
    sub_cov = sub_cov + ridge_lambda * np.eye(n)
    
    try:
        precision = np.linalg.inv(sub_cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(sub_cov)
    
    p_xx, p_yy, p_xy = precision[0, 0], precision[1, 1], precision[0, 1]
    if p_xx < 1e-10 or p_yy < 1e-10:
        return 0.0
    
    r = -p_xy / np.sqrt(p_xx * p_yy)
    return float(np.clip(r, -0.999999, 0.999999))

fz.partial_corr_coef = partial_corr_coef

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_outcome():
    """Load survival outcome"""
    outcome_file = os.path.join(FILTERED_DIR, 'outcome.csv')
    outcome_df = pd.read_csv(outcome_file, index_col=0)
    return outcome_df

def load_dataset(filename):
    """Load Methylation dataset"""
    df = pd.read_csv(os.path.join(FILTERED_DIR, filename), index_col=0)
    original_features = df.shape[1]
    print(f"   Using all {df.shape[1]:,} RNA features")
    return df, original_features

def preprocess_data(rna_df, outcome):
    """Prepare RNA data for MB algorithms"""
    # RNA is already normalized (Log2 + Z-score from preprocessing)
    # Just add small noise for numerical stability
    X = rna_df.values.astype(float)
    X = X + np.random.normal(0.0, 1e-8, size=X.shape)
    
    y = outcome['OS.time'].values.astype(float)
    full_data = np.column_stack([X, y]).astype(float)
    target_idx = X.shape[1]
    
    return full_data, target_idx

def run_mb_algorithm(algorithm_name, data_matrix, target_idx, alpha=0.05):
    """Run single MB algorithm"""
    t0 = time.time()
    
    try:
        if algorithm_name == 'IAMB':
            result = IAMB(
                data=data_matrix,
                target=target_idx,
                is_discrete=False,
                alaph=alpha
            )
        elif algorithm_name == 'GSMB':
            result = GSMB(
                data=data_matrix,
                target=target_idx,
                is_discrete=False,
                alaph=alpha
            )
        elif algorithm_name == 'MMMB':
            result = MMMB(
                data=data_matrix,
                target=target_idx,
                is_discrete=False,
                alaph=alpha
            )
        
        elapsed = time.time() - t0
        
        if isinstance(result, tuple):
            mb_idx = list(result[0])
        else:
            mb_idx = list(result) if result is not None else []
        
        mb_idx = [i for i in mb_idx if i != target_idx and 0 <= i < data_matrix.shape[1]-1]
        
        return mb_idx, elapsed, None
        
    except Exception as e:
        elapsed = time.time() - t0
        return [], elapsed, str(e)

def compute_cindex(rna_df, outcome, gene_names):
    """Compute C-index for selected genes"""
    if len(gene_names) == 0:
        return 0.0
    
    try:
        # Get selected features
        X_sel = rna_df[gene_names].values
        
        # Simple risk score: sum of features
        risk_scores = X_sel.sum(axis=1)
        
        # Compute C-index
        c_index = concordance_index(
            event_times=outcome['OS.time'].values,
            predicted_scores=risk_scores,
            event_observed=outcome['OS'].values
        )
        
        return float(c_index)
    except Exception as e:
        print(f"   Warning: C-index computation failed: {e}")
        return 0.0

def evaluate_genes(rna_df, outcome_series, gene_names):
    """Compute MI scores and C-index"""
    if len(gene_names) == 0:
        return {
            'mean_mi': 0.0,
            'max_mi': 0.0,
            'min_mi': 0.0,
            'c_index': 0.0,
            'mi_scores': []
        }
    
    # Mutual Information
    X_sel = rna_df[gene_names]
    mi_scores = mutual_info_regression(X_sel, outcome_series, random_state=42, n_neighbors=5)
    
    # C-index
    try:
        outcome_file = os.path.join(FILTERED_DIR, 'outcome.csv')
        outcome_df = pd.read_csv(outcome_file, index_col=0)
        c_index = compute_cindex(rna_df, outcome_df, gene_names)
    except:
        c_index = 0.0
    
    return {
        'mean_mi': float(np.mean(mi_scores)),
        'max_mi': float(np.max(mi_scores)),
        'min_mi': float(np.min(mi_scores)),
        'c_index': float(c_index),
        'mi_scores': mi_scores.tolist()
    }

def compute_consensus(algo_results, feature_names, algorithms_used):
    """Compute consensus between algorithms"""
    sets = {}
    for algo in algorithms_used:
        if algo in algo_results:
            indices = algo_results[algo]['indices']
            sets[algo] = set([feature_names[i] for i in indices])
        else:
            sets[algo] = set()
    
    # All agree
    if len(sets) > 0:
        all_agree = set.intersection(*sets.values())
    else:
        all_agree = set()
    
    # Any 2 agree
    any2 = set()
    algo_list = list(sets.keys())
    if len(algo_list) >= 2:
        for i in range(len(algo_list)):
            for j in range(i+1, len(algo_list)):
                any2 |= (sets[algo_list[i]] & sets[algo_list[j]])
    
    # Union
    union_all = set()
    for s in sets.values():
        union_all |= s
    
    return {
        'all_agree': sorted(list(all_agree)),
        'any2': sorted(list(any2)),
        'union': sorted(list(union_all))
    }

def process_single_config(dataset_name, rna_df, outcome, algorithm, alpha, pbar, original_features):
    """Process single configuration"""
    feature_names = rna_df.columns.tolist()
    
    data_matrix, target_idx = preprocess_data(rna_df, outcome)
    
    pbar.set_description(f"{dataset_name[:30]:30s} | {algorithm:10s} | α={alpha}")
    indices, elapsed, error = run_mb_algorithm(algorithm, data_matrix, target_idx, alpha)
    
    if error:
        print(f"\n   Error: {error}")
        return None
    
    causal_probes = [feature_names[i] for i in indices]
    # Fallback: if MB selected fewer than MB_MIN_FEATURES, pad with composite-ranked features
    if len(causal_probes) < MB_MIN_FEATURES:
        _cache = os.path.join(FILTERED_DIR, '..', 'meth_statistics_cache.csv')
        if os.path.exists(_cache):
            _stats = pd.read_csv(_cache)
            _existing = set(causal_probes)
            _pool = [g for g in _stats['probe'].tolist()
                     if g in feature_names and g not in _existing]
            _needed = MB_MIN_FEATURES - len(causal_probes)
            causal_probes = causal_probes + _pool[:_needed]
            print(f"   Padded to {len(causal_probes)} features (composite fallback)")


    metrics = evaluate_genes(rna_df, outcome['OS.time'], causal_probes)
    
    print(f"\n   {algorithm:10s} α={alpha}: {len(causal_probes):3d} genes, MI={metrics['mean_mi']:.4f}, C-index={metrics['c_index']:.4f}, {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    return {
        'dataset': dataset_name,
        'algorithm': algorithm,
        'alpha': alpha,
        'n_original_features': original_features,
        'n_input_features': len(feature_names),
        'n_causal_probes': len(causal_probes),
        'mean_mi': metrics['mean_mi'],
        'max_mi': metrics['max_mi'],
        'min_mi': metrics['min_mi'],
        'c_index': metrics['c_index'],
        'time_sec': elapsed,
        'causal_probes': causal_probes
    }

def process_dataset(dataset_file, outcome):
    """Process single dataset"""
    dataset_name = dataset_file.replace('.csv', '')
    
    print("\n" + "="*80)
    print(f"Dataset: {dataset_name}")
    print("="*80)
    
    rna_df, original_features = load_dataset(dataset_file)
    print(f"Shape: {rna_df.shape}")
    
    dataset_output = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_output, exist_ok=True)
    
    algo_results_by_alpha = {}
    all_results = []
    
    total_configs = sum(len(ALGORITHMS_BY_ALPHA[alpha]) for alpha in ALPHAS)
    
    with tqdm(total=total_configs, desc=dataset_name[:30]) as pbar:
        for alpha in ALPHAS:
            algo_results_by_alpha[alpha] = {}
            algorithms = ALGORITHMS_BY_ALPHA[alpha]
            
            for algo in algorithms:
                result = process_single_config(dataset_name, rna_df, outcome, algo, alpha, pbar, original_features)
                
                if result:
                    all_results.append(result)
                    algo_results_by_alpha[alpha][algo] = {
                        'indices': [rna_df.columns.get_loc(g) for g in result['causal_probes']],
                        'genes': result['causal_probes']
                    }
                    
                    genes_file = os.path.join(dataset_output, f'{algo}_alpha{alpha}_genes.txt')
                    with open(genes_file, 'w') as f:
                        f.write('\n'.join(result['causal_probes']))
                    
                    metrics_file = os.path.join(dataset_output, f'{algo}_alpha{alpha}_metrics.json')
                    with open(metrics_file, 'w') as f:
                        json.dump(result, f, indent=2)
                
                pbar.update(1)
    
    print(f"\nConsensus Analysis:")
    
    for alpha in ALPHAS:
        if alpha in algo_results_by_alpha:
            algorithms = ALGORITHMS_BY_ALPHA[alpha]
            consensus = compute_consensus(algo_results_by_alpha[alpha], rna_df.columns.tolist(), algorithms)
            
            print(f"\n  Alpha = {alpha} ({len(algorithms)} algorithms):")
            print(f"    All {len(algorithms)} agree: {len(consensus['all_agree']):3d} genes")
            if len(algorithms) >= 2:
                print(f"    Any 2 agree:     {len(consensus['any2']):3d} genes")
            print(f"    Union (any):     {len(consensus['union']):3d} genes")
            
            if len(consensus['all_agree']) > 0:
                cons_file = os.path.join(dataset_output, f'consensus_all_alpha{alpha}.txt')
                with open(cons_file, 'w') as f:
                    f.write('\n'.join(consensus['all_agree']))
            
            if len(consensus['any2']) > 0:
                cons_file = os.path.join(dataset_output, f'consensus_any2_alpha{alpha}.txt')
                with open(cons_file, 'w') as f:
                    f.write('\n'.join(consensus['any2']))
            
            for result in all_results:
                if result['alpha'] == alpha:
                    result['consensus_all'] = len(consensus['all_agree'])
                    result['consensus_any2'] = len(consensus['any2'])
                    result['consensus_union'] = len(consensus['union'])
    
    return all_results

# ============================================================================
# MAIN
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

outcome = load_outcome()
print(f"Outcome: {len(outcome)} samples")

# FILTER OUT NON-DATASET FILES
dataset_files = [
    f for f in os.listdir(FILTERED_DIR) 
    if f.endswith('.csv') 
    and 'summary' not in f.lower()
    and 'outcome' not in f.lower()
    and 'annotated' not in f.lower()
    and 'statistics' not in f.lower()
    and f.startswith('meth_')
    and 'probes.csv' in f
]

print(f"Found {len(dataset_files)} datasets:")
for f in dataset_files:
    print(f"   - {f}")

all_results = []

print("\n" + "="*80)
print("PROCESSING METHYLATION DATASETS")
print("="*80)

for idx, dataset_file in enumerate(dataset_files, 1):
    print(f"\n[{idx}/{len(dataset_files)}] Starting: {dataset_file}")
    
    try:
        results = process_dataset(dataset_file, outcome)
        all_results.extend(results)
        print(f"[{idx}/{len(dataset_files)}] Completed: {dataset_file}")
        
    except Exception as e:
        print(f"[{idx}/{len(dataset_files)}] FAILED: {dataset_file}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

if all_results:
    summary_df = pd.DataFrame(all_results)
    display_df = summary_df.drop(columns=['causal_probes'], errors='ignore')
    
    print("\nAll Results:")
    print(display_df.to_string(index=False))
    
    summary_file = os.path.join(OUTPUT_DIR, 'summary_all_results.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"\nBy Algorithm:")
    for algo in sorted(summary_df['algorithm'].unique()):
        algo_df = summary_df[summary_df['algorithm'] == algo]
        print(f"  {algo:10s}: avg {algo_df['n_causal_probes'].mean():.1f} genes, "
              f"MI={algo_df['mean_mi'].mean():.4f}, C-index={algo_df['c_index'].mean():.4f}, time={algo_df['time_sec'].mean():.1f}s")
    
    print(f"\nBy Alpha:")
    for alpha in ALPHAS:
        alpha_df = summary_df[summary_df['alpha'] == alpha]
        if len(alpha_df) > 0:
            print(f"  α={alpha}: avg {alpha_df['n_causal_probes'].mean():.1f} genes, "
                  f"C-index={alpha_df['c_index'].mean():.4f}, consensus(all)={alpha_df['consensus_all'].mean():.1f}")
    
    print(f"\nConsensus Summary:")
    print(f"  Avg all agree: {summary_df['consensus_all'].mean():.1f} genes")
    print(f"  Avg any 2 agree: {summary_df['consensus_any2'].mean():.1f} genes")
    print(f"  Max all agree: {summary_df['consensus_all'].max():.0f} genes")
    
    print(f"\n💾 Summary saved: {summary_file}")
else:
    print("\n⚠️  No results to summarize")

print("\n" + "="*80)
print("METHYLATION PRODUCTION RUN COMPLETE!")
print(f"Results: {OUTPUT_DIR}")
print("="*80)