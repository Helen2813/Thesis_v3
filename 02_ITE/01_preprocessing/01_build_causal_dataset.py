"""
build_causal_dataset.py
=======================
Read best_strategy_features.txt, extract those columns from 08_composite.csv,
merge with outcome.csv (OS.time, OS),
save causal_features_dataset.csv.
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# paths relative to this script
HERE         = Path(__file__).resolve().parent   # 02_ITE/01_preprocessing/
ROOT         = HERE.parents[1]                   # Thesis_v3/

FEATURES_TXT = ROOT / "01_Causal_feature_extraction" / "MB" / "results_fallback_experiment" / "best_strategy_features.txt"
COMPOSITE    = ROOT / "01_Causal_feature_extraction" / "MERGE_continuous_outer" / "08_composite.csv"
OUTCOME      = ROOT / "data" / "outcome.csv"
OUTPUT_DIR   = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SURVIVAL_TIME  = "OS.time"
SURVIVAL_EVENT = "OS"

KNOWN_PREFIXES = {"CLIN", "RNA", "CNV", "MUT", "PROT", "METH", "MIRNA"}

# ── 1. load feature list ───────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Load best_strategy_features.txt")
print("=" * 60)

with open(FEATURES_TXT, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

selected = [l for l in lines if l.split("_")[0] in KNOWN_PREFIXES]
print(f"Features found: {len(selected)}")
for i, feat in enumerate(selected, 1):
    print(f"  {i:2d}. {feat}")

# ── 2. load composite dataset ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 2 — Load composite dataset")
print("=" * 60)

df = pd.read_csv(COMPOSITE, index_col=0)
df.index = df.index.astype(str).str.strip()
print(f"Shape: {df.shape}")

missing = [f for f in selected if f not in df.columns]
found   = [f for f in selected if f in df.columns]

if missing:
    print(f"\nWARNING: {len(missing)} features not found in composite:")
    for f in missing:
        print(f"  {f}")
else:
    print(f"All {len(found)} features found.")

# ── 3. load outcome ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 3 — Load outcome")
print("=" * 60)

outcome = pd.read_csv(OUTCOME, index_col=0)
outcome.index = outcome.index.astype(str).str.strip().str[:12]
print(f"Outcome shape:   {outcome.shape}")
print(f"Outcome columns: {outcome.columns.tolist()}")

assert SURVIVAL_TIME  in outcome.columns, f"{SURVIVAL_TIME} not found in outcome.csv"
assert SURVIVAL_EVENT in outcome.columns, f"{SURVIVAL_EVENT} not found in outcome.csv"

# index diagnostics
print(f"\nComposite index sample: {df.index[:3].tolist()}")
print(f"Outcome index sample:   {outcome.index[:3].tolist()}")
overlap = df.index.intersection(outcome.index)
print(f"Overlapping patients:   {len(overlap)}")

# ── 4. build dataset ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 4 — Build dataset")
print("=" * 60)

result = df[found].join(outcome[[SURVIVAL_TIME, SURVIVAL_EVENT]], how="inner")

print(f"Shape:   {result.shape}")
print(f"Samples: {len(result)}")
print(f"Events:  {int(result[SURVIVAL_EVENT].sum())}")

mv = result[found].isnull().sum()
mv = mv[mv > 0]
if len(mv):
    print(f"\nMissing values:")
    print(mv.to_string())
else:
    print("Missing values: none")

mods = Counter(c.split("_")[0] for c in found)
print(f"\nModalities:")
for mod, n in sorted(mods.items()):
    print(f"  {mod}: {n}")

# ── 5. save ────────────────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / "causal_features_dataset.csv"
result.to_csv(out_path)
print(f"\nSaved: {out_path}")
