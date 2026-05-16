"""
run_pipeline.py
---------------
Master script that executes the full SparseRank pipeline end-to-end.

Usage:
  cd sparserank/
  python run_pipeline.py

Steps:
  1. Load (or generate) dataset
  2. L1 feature selection sweep
  3. Anomaly detection with IF on sparse features
  4. SHAP explainability
  5. Evaluation + figures
"""

import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

t_start = time.time()

print("=" * 60)
print("  SparseRank — Full Pipeline")
print("=" * 60)

# ── Step 1: Data ──────────────────────────────────────────────────────────────
print("\n[STEP 1] Load / generate dataset")
from src.data_loader import load_dataset, train_test_split_df
df = load_dataset()
print(f"  Dataset: {df.shape[0]} windows × {df.shape[1]-3} features")

# ── Step 2: Feature selection ─────────────────────────────────────────────────
print("\n[STEP 2] L1-regularized feature selection")
import src.feature_select as fs
selected_idx, feat_cols, scaler = fs.main()

# ── Step 3: Anomaly detection ─────────────────────────────────────────────────
print("\n[STEP 3] Isolation Forest on sparse features")
import src.anomaly_detector as ad
ad.main()

# ── Step 4: SHAP explainability ───────────────────────────────────────────────
print("\n[STEP 4] SHAP explainability")
import src.shap_explainer as se
shap_summary = se.main()

# ── Step 5: Evaluation & figures ──────────────────────────────────────────────
print("\n[STEP 5] Evaluation + figures")
import src.evaluate as ev
ev.main()

# ── Timing ────────────────────────────────────────────────────────────────────
t_total = time.time() - t_start
print(f"\n{'='*60}")
print(f"  Pipeline complete in {t_total:.1f}s")
print(f"{'='*60}")
print("\nOutputs:")
print("  data/processed/feature_select_results.csv")
print("  data/processed/eval_metrics.csv")
print("  data/processed/shap_feature_importance.csv")
print("  data/processed/baseline_comparison.csv")
print("  experiments/figures/sparserank_all_figures.png")
