"""
Create Treatment-Specific Arms from clinical.tsv
==================================================
Thesis: Causal Multimodal Analysis of Breast Cancer Survival (TCGA-BRCA)

Reads treatments.treatment_type from clinical.tsv (already in the project)
and creates per-patient binary treatment indicators:

  T_hormone          — received Hormone Therapy (tamoxifen, AIs, etc.)
  T_chemo            — received Chemotherapy
  T_targeted         — received Targeted Molecular Therapy (trastuzumab, etc.)
  T_hormone_excl     — received Hormone Therapy AND NOT Chemotherapy AND NOT Targeted
  T_any              — received any treatment (original T column, preserved)

Saves:
  ite_ready_dataset_v2.csv   — updated dataset with treatment arm columns

Run from anywhere inside Thesis_v3/:
  python 04_create_treatment_arms.py
"""

import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
BASE_DIR   = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]
     if (p / '02_ITE' / '01_preprocessing' / 'output' / 'ite_ready_dataset.csv').exists()),
    None
)
if BASE_DIR is None:
    print("ERROR: Cannot find ite_ready_dataset.csv — run from inside Thesis_v3/")
    sys.exit(1)

# clinical.tsv — try multiple known locations
CLINICAL_CANDIDATES = [
    BASE_DIR / 'data' / 'drags' / 'clinical.tsv',
    BASE_DIR / 'data' / 'clinical.tsv',
    BASE_DIR / 'clinical.tsv',
]
CLINICAL_PATH = next((p for p in CLINICAL_CANDIDATES if p.exists()), None)

ITE_PATH   = BASE_DIR / '02_ITE' / '01_preprocessing' / 'output' / 'ite_ready_dataset.csv'
OUTPUT_DIR = BASE_DIR / '02_ITE' / '01_preprocessing' / 'output'

print("=" * 65)
print("  CREATE TREATMENT ARMS FROM clinical.tsv")
print("=" * 65)
print(f"  Base    : {BASE_DIR}")
print(f"  ITE     : {ITE_PATH}")
print(f"  Clinical: {CLINICAL_PATH or 'NOT FOUND'}")

if CLINICAL_PATH is None:
    print("\n  ERROR: clinical.tsv not found. Searched:")
    for p in CLINICAL_CANDIDATES:
        print(f"    {p}")
    print("\n  Copy clinical.tsv to one of the above locations and re-run.")
    sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────────
TCGA_NA = ["'--","--","NA","N/A","Not Reported","not reported",
           "[Not Available]","[Unknown]","[Not Applicable]","nan",""]

print("\nLoading clinical.tsv...")
clin = pd.read_csv(CLINICAL_PATH, sep='\t', na_values=TCGA_NA, low_memory=False)
print(f"  Shape: {clin.shape}")

print("\nLoading ite_ready_dataset.csv...")
ite = pd.read_csv(ITE_PATH)
print(f"  Shape: {ite.shape}")
print(f"  Patients: {ite['patient_id'].nunique()}")

# ── Normalize patient IDs ──────────────────────────────────────────────────────
clin['patient_id'] = (clin['cases.submitter_id']
                      .astype(str).str.strip().str.upper().str[:12])
ite['patient_id_norm'] = (ite['patient_id']
                          .astype(str).str.strip().str.upper().str[:12])

print(f"\n  clinical.tsv unique patients: {clin['patient_id'].nunique()}")
print(f"  ite_ready unique patients   : {ite['patient_id_norm'].nunique()}")
print(f"  Overlap                     : {len(set(clin['patient_id']) & set(ite['patient_id_norm']))}")

# ── Extract treatment types per patient ───────────────────────────────────────
ttype = clin[['patient_id', 'treatments.treatment_type',
              'treatments.therapeutic_agents']].copy()

def patients_with_type(keyword):
    mask = ttype['treatments.treatment_type'].str.lower().str.contains(
        keyword, na=False)
    return set(ttype[mask]['patient_id'])

hormone_pts  = patients_with_type('hormone')
chemo_pts    = patients_with_type('chemo')
targeted_pts = patients_with_type('targeted')
radiation_pts= patients_with_type('radiation')

# Exclusive hormone: hormone but NOT chemo AND NOT targeted
hormone_excl_pts = hormone_pts - chemo_pts - targeted_pts

print(f"\n  Treatment type breakdown:")
print(f"    Hormone Therapy  : {len(hormone_pts)} patients")
print(f"    Chemotherapy     : {len(chemo_pts)} patients")
print(f"    Targeted Therapy : {len(targeted_pts)} patients")
print(f"    Radiation        : {len(radiation_pts)} patients")
print(f"\n    Hormone AND Chemo: {len(hormone_pts & chemo_pts)}")
print(f"    Hormone ONLY     : {len(hormone_excl_pts)}  (no chemo, no targeted)")
print(f"    Chemo ONLY       : {len(chemo_pts - hormone_pts - targeted_pts)}")

# ── Merge into ite_ready_dataset ───────────────────────────────────────────────
df = ite.copy()
pid = df['patient_id_norm']

df['T_hormone']      = pid.isin(hormone_pts).astype(int)
df['T_chemo']        = pid.isin(chemo_pts).astype(int)
df['T_targeted']     = pid.isin(targeted_pts).astype(int)
df['T_radiation']    = pid.isin(radiation_pts).astype(int)
df['T_hormone_excl'] = pid.isin(hormone_excl_pts).astype(int)

# Clean up helper column
df = df.drop(columns=['patient_id_norm'])

print(f"\n  Merged treatment arms into ite_ready_dataset:")
for col in ['T_any', 'T_hormone', 'T_hormone_excl', 'T_chemo', 'T_targeted']:
    tcol = 'T' if col == 'T_any' else col
    if tcol in df.columns:
        n = df[tcol].sum()
        print(f"    {col:<20}: {n} ({n/len(df)*100:.1f}%)")

# ── Receptor subgroup cross-tabulation ────────────────────────────────────────
er   = (df['ER_status'].fillna(0)   > 0.5).astype(int)
pr   = (df['PR_status'].fillna(0)   > 0.5).astype(int)
her2 = (df['HER2_status'].fillna(0) > 0.5).astype(int)

print(f"\n  Hormone arm validation (ER+/PR+ HER2- subgroup):")
mask_horm = ((er | pr) & (her2 == 0))
sub = df[mask_horm]
print(f"    ER+/PR+ HER2- patients   : {mask_horm.sum()}")
print(f"    Received Hormone Therapy : {sub['T_hormone'].sum()} ({sub['T_hormone'].mean()*100:.1f}%)")
print(f"    Received Hormone EXCL    : {sub['T_hormone_excl'].sum()} ({sub['T_hormone_excl'].mean()*100:.1f}%)")
print(f"    Received Chemotherapy    : {sub['T_chemo'].sum()} ({sub['T_chemo'].mean()*100:.1f}%)")

print(f"\n  TNBC subgroup:")
mask_tnbc = ((er == 0) & (pr == 0) & (her2 == 0))
sub_tnbc = df[mask_tnbc]
print(f"    TNBC patients            : {mask_tnbc.sum()}")
print(f"    Received Chemotherapy    : {sub_tnbc['T_chemo'].sum()} ({sub_tnbc['T_chemo'].mean()*100:.1f}%)")
print(f"    Received Hormone Therapy : {sub_tnbc['T_hormone'].sum()} ({sub_tnbc['T_hormone'].mean()*100:.1f}%)")

print(f"\n  HER2+ subgroup:")
mask_her2 = (her2 == 1)
sub_her2 = df[mask_her2]
print(f"    HER2+ patients           : {mask_her2.sum()}")
print(f"    Received Targeted        : {sub_her2['T_targeted'].sum()} ({sub_her2['T_targeted'].mean()*100:.1f}%)")
print(f"    Received Chemotherapy    : {sub_her2['T_chemo'].sum()} ({sub_her2['T_chemo'].mean()*100:.1f}%)")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / 'ite_ready_dataset_v2.csv'
df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")
print(f"  Shape: {df.shape}")
print(f"  New columns: T_hormone, T_chemo, T_targeted, T_radiation, T_hormone_excl")

# Update metadata
meta_path = OUTPUT_DIR / 'preprocessing_metadata.json'
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    meta['treatment_arms'] = {
        'T':              'any treatment (original)',
        'T_hormone':      'Hormone Therapy from clinical.tsv',
        'T_hormone_excl': 'Hormone Therapy ONLY (no chemo, no targeted)',
        'T_chemo':        'Chemotherapy from clinical.tsv',
        'T_targeted':     'Targeted Molecular Therapy from clinical.tsv',
        'T_radiation':    'Radiation from clinical.tsv',
        'source_file':    str(CLINICAL_PATH),
    }
    with open(OUTPUT_DIR / 'preprocessing_metadata_v2.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: preprocessing_metadata_v2.json")

print(f"\n{'='*65}")
print("  NEXT STEPS:")
print("  1. Run ablation_study.py with ite_ready_dataset_v2.csv")
print("     -> change INPUT_DIR path to load _v2 dataset")
print("  2. Run final_ite_comparison.py with _v2")
print("     -> T_hormone replaces T_bin for hormone arm")
print("  3. Re-run hormone investigation with proper T_hormone")
print(f"{'='*65}")
print("  DONE")
