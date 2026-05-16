"""
shap_explainer.py
-----------------
Trains a surrogate DecisionTree on IF-labeled anomaly windows,
computes TreeSHAP values, and produces:

  1. Per-feature SHAP importance S(feature_i) for the L1-selected set
  2. Comparison with full-feature SHAP ranking (Spearman ρ)
  3. Top-20 business-critical features ranked by sparse SHAP

Outputs:
  - data/processed/sparse_shap_values.npy
  - data/processed/shap_feature_importance.csv   (top-20)
  - data/processed/shap_rank_comparison.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

import shap
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr

from src.data_loader import load_dataset, train_test_split_df


def compute_shap(X_subset, y, feature_names_subset, label="sparse"):
    """
    Trains a surrogate DecisionTree (depth=5) and computes SHAP values.
    Returns (shap_values, per_feature_importance).
    """
    clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X_subset, y)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_subset)

    # sv shape handling for different shap versions
    if isinstance(sv, list):
        sv = sv[1]   # class=1 (anomaly) SHAP values
    elif hasattr(sv, 'values'):
        sv = sv.values

    sv = np.array(sv)
    # (n_samples, n_features, n_classes) → take class=1
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    # (n_samples, n_features) → ok

    per_feat = np.abs(sv).mean(axis=0).flatten()

    importance_df = pd.DataFrame({
        "feature": list(feature_names_subset),
        f"shap_mean_abs_{label}": list(per_feat)
    }).sort_values(f"shap_mean_abs_{label}", ascending=False).reset_index(drop=True)

    print(f"[shap_explainer] Top-5 features ({label}):")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:50s}  {row[f'shap_mean_abs_{label}']:.6f}")

    return sv, importance_df


def main():
    print("[shap_explainer] Loading dataset and artifacts...")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te, _, _, feat_cols = train_test_split_df(df)

    selected_idx = np.load("data/processed/selected_idx.npy")
    with open("data/processed/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    y_pred = np.load("data/processed/anomaly_preds.npy")
    k = len(selected_idx)

    X_te_sc_full = scaler.transform(X_te)
    X_te_sc_sparse = X_te_sc_full[:, selected_idx]

    feat_names_sparse = [feat_cols[i] for i in selected_idx]
    feat_names_full   = list(feat_cols)

    print(f"  Sparse feature set: k={k}")
    print(f"  Test windows: {len(y_te)} (anomaly: {y_te.sum()})")

    # ── SHAP on sparse features ───────────────────────────────────────────────
    print("\n[shap_explainer] Computing SHAP on L1-selected (sparse) features...")
    sv_sparse, imp_sparse = compute_shap(X_te_sc_sparse, y_pred, feat_names_sparse, "sparse")

    # ── SHAP on full features ─────────────────────────────────────────────────
    print("\n[shap_explainer] Computing SHAP on full feature set for comparison...")
    sv_full, imp_full = compute_shap(X_te_sc_full, y_pred, feat_names_full, "full")

    # ── Spearman rank correlation on shared features ──────────────────────────
    print("\n[shap_explainer] Computing Spearman ρ on shared feature rankings...")

    # Merge on shared features
    shared = set(feat_names_sparse) & set(feat_names_full)
    sparse_ranks = imp_sparse.reset_index(drop=True)
    sparse_ranks["rank_sparse"] = sparse_ranks.index + 1

    full_ranks = imp_full.reset_index(drop=True)
    full_ranks["rank_full"] = full_ranks.index + 1

    merged = sparse_ranks[["feature", "shap_mean_abs_sparse", "rank_sparse"]].merge(
        full_ranks[["feature", "shap_mean_abs_full", "rank_full"]],
        on="feature"
    )

    if len(merged) > 1:
        rho, pval = spearmanr(merged["rank_sparse"], merged["rank_full"])
        print(f"  Shared features: {len(merged)}")
        print(f"  Spearman ρ(sparse_rank, full_rank) = {rho:.4f}  p={pval:.4e}")
        if rho > 0.75:
            print("  ✓ PASS: Sparse SHAP ranking faithfully represents full model")
        else:
            print(f"  ⚠ rho={rho:.4f} < 0.75 — rankings diverge")
    else:
        rho, pval = 0.0, 1.0
        print("  ⚠ Not enough shared features for ranking comparison")

    merged["spearman_rho"] = rho
    merged["spearman_pval"] = pval

    # ── Attribution coverage ──────────────────────────────────────────────────
    total_attr_full   = imp_full[f"shap_mean_abs_full"].sum()
    attr_from_sparse  = imp_full[imp_full["feature"].isin(feat_names_sparse)][f"shap_mean_abs_full"].sum()
    if total_attr_full > 0:
        coverage_pct = attr_from_sparse / total_attr_full * 100
    else:
        coverage_pct = 0.0
    print(f"\n[shap_explainer] Attribution coverage:")
    print(f"  L1-selected features ({k}/{len(feat_cols)}) account for "
          f"{coverage_pct:.1f}% of total SHAP attribution")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save("data/processed/sparse_shap_values.npy", sv_sparse)
    imp_sparse.to_csv("data/processed/shap_feature_importance.csv", index=False)
    merged.to_csv("data/processed/shap_rank_comparison.csv", index=False)

    # Summary dict
    summary = {
        "k_sparse": k,
        "p_full": len(feat_cols),
        "spearman_rho_rank": round(rho, 4),
        "attribution_coverage_pct": round(coverage_pct, 1),
        "top1_feature": imp_sparse.iloc[0]["feature"],
        "top5_features": imp_sparse.head(5)["feature"].tolist()
    }
    pd.DataFrame([summary]).to_csv("data/processed/shap_summary.csv", index=False)
    print(f"\n[shap_explainer] Top feature: {summary['top1_feature']}")
    print(f"[shap_explainer] Saved SHAP results to data/processed/")

    return summary


if __name__ == "__main__":
    os.chdir("/home/claude/sparserank")
    main()
