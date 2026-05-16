"""
feature_select.py
-----------------
L1-regularized feature selection for SparseRank.

Sweeps LogisticRegression(penalty='l1') over C ∈ {0.001, 0.01, 0.1, 1.0}.
For each C, trains IsolationForest on the selected k features and records F1.
Also sweeps three anomaly-ratio train splits for stability estimation.

Outputs:
  - data/processed/selected_idx.npy   : final selected feature indices
  - data/processed/feature_select_results.csv : full sweep results table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_dataset, train_test_split_df

C_VALUES = [0.001, 0.01, 0.1, 1.0]
ANOMALY_RATIOS = [0.05, 0.10, 0.40]   # test contamination fractions for IF
IF_SEED = 42


def run_l1_sweep(X_tr, X_te, y_tr, y_te, scaler):
    """Run L1 sweep over C values. Returns list of result dicts."""
    results = []
    for C in C_VALUES:
        lr = LogisticRegression(
            penalty='l1', C=C, solver='saga',
            max_iter=2000, random_state=42, class_weight='balanced'
        )
        lr.fit(scaler.transform(X_tr), y_tr)
        coefs = lr.coef_[0]
        idx = np.where(np.abs(coefs) > 1e-8)[0]
        k = len(idx)

        if k == 0:
            results.append({"C": C, "k": 0, "F1": 0.0, "Precision": 0.0,
                             "Recall": 0.0, "cost_reduction_pct": 100.0})
            continue

        # Train IF on selected features only
        X_tr_sub = scaler.transform(X_tr)[:, idx]
        X_te_sub = scaler.transform(X_te)[:, idx]

        if_ = IsolationForest(
            n_estimators=100,
            contamination=min(float(y_tr.mean()), 0.49),
            random_state=IF_SEED
        )
        if_.fit(X_tr_sub)
        scores = if_.decision_function(X_te_sub)
        y_pred = (scores < 0).astype(int)

        f1 = f1_score(y_te, y_pred, zero_division=0)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        cost_red = (1 - k / X_tr.shape[1]) * 100

        results.append({
            "C": C, "k": k, "F1": round(f1, 4),
            "Precision": round(prec, 4), "Recall": round(rec, 4),
            "cost_reduction_pct": round(cost_red, 1)
        })
        print(f"  C={C:<6} k={k:3d}  F1={f1:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  cost_red={cost_red:.1f}%")
    return results


def run_full_feature_baseline(X_tr, X_te, y_tr, y_te, scaler):
    """Full-feature IF baseline for comparison."""
    if_ = IsolationForest(
        n_estimators=100,
        contamination=min(float(y_tr.mean()), 0.49),
        random_state=IF_SEED
    )
    if_.fit(scaler.transform(X_tr))
    scores = if_.decision_function(scaler.transform(X_te))
    y_pred = (scores < 0).astype(int)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    print(f"  C=FULL   k={X_tr.shape[1]:3d}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  cost_red=0.0%")
    return {"C": "FULL", "k": X_tr.shape[1], "F1": round(f1, 4),
            "Precision": round(prec, 4), "Recall": round(rec, 4),
            "cost_reduction_pct": 0.0}


def stability_sweep(X, y, feature_names):
    """
    Run L1 feature count across 3 anomaly-ratio splits → report mean k.
    """
    print("\n[feature_select] Stability sweep over 3 anomaly ratios...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(sss.split(X, y))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    scaler.fit(X_tr)
    stability = []
    for C in [0.01]:  # use the target C only for stability
        for ar in ANOMALY_RATIOS:
            # Simulate higher anomaly ratio by subsampling anomaly windows
            pos_idx = np.where(y_tr == 1)[0]
            neg_idx = np.where(y_tr == 0)[0]
            n_desired_pos = int(len(neg_idx) * ar / (1 - ar))
            n_desired_pos = min(n_desired_pos, len(pos_idx))
            rng = np.random.default_rng(42)
            chosen_pos = rng.choice(pos_idx, n_desired_pos, replace=False)
            X_sub = np.vstack([X_tr[neg_idx], X_tr[chosen_pos]])
            y_sub = np.hstack([y_tr[neg_idx], y_tr[chosen_pos]])

            lr = LogisticRegression(
                penalty='l1', C=C, solver='saga',
                max_iter=2000, random_state=42, class_weight='balanced'
            )
            lr.fit(scaler.transform(X_sub), y_sub)
            k = int((np.abs(lr.coef_[0]) > 1e-8).sum())
            stability.append({"C": C, "anomaly_ratio": ar, "k": k})
            print(f"  C={C}  anomaly_ratio={ar:.0%}  k={k}")
    return stability


def main():
    print("[feature_select] Loading dataset...")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te, _, _, feat_cols = train_test_split_df(df)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"  p={X_tr.shape[1]} total features")

    scaler = StandardScaler()
    scaler.fit(X_tr)

    print("\n[feature_select] L1 sweep (C ∈ {0.001, 0.01, 0.1, 1.0}):")
    results = run_l1_sweep(X_tr, X_te, y_tr, y_te, scaler)
    full_result = run_full_feature_baseline(X_tr, X_te, y_tr, y_te, scaler)

    all_results = results + [full_result]
    df_res = pd.DataFrame(all_results)
    df_res.to_csv("data/processed/feature_select_results.csv", index=False)
    print(f"\n[feature_select] Results saved to data/processed/feature_select_results.csv")

    # ── Select best C by: ≥95% of full-feature F1, minimum k ──────────────────
    full_f1 = full_result["F1"]
    threshold = full_f1 * 0.95
    candidates = df_res[df_res["C"] != "FULL"].copy()
    candidates["k_num"] = candidates["k"].astype(int)
    passing = candidates[candidates["F1"] >= threshold].sort_values("k_num")

    if len(passing) == 0:
        # Fall back to C giving best F1
        best_row = candidates.sort_values("F1", ascending=False).iloc[0]
        print(f"\n[feature_select] ⚠ No C achieves 95% of full-feature F1. "
              f"Using best: C={best_row.C}, k={best_row.k}, F1={best_row.F1}")
    else:
        best_row = passing.iloc[0]
        print(f"\n[feature_select] ✓ Best C={best_row.C}: k={best_row.k} features, "
              f"F1={best_row.F1:.4f} vs full={full_f1:.4f} "
              f"({best_row.F1/full_f1*100:.1f}%)")
        print(f"  → Cost reduction: {best_row.cost_reduction_pct:.1f}%")

    # Save selected indices
    best_C = float(best_row.C)
    lr_final = LogisticRegression(
        penalty='l1', C=best_C, solver='saga',
        max_iter=2000, random_state=42, class_weight='balanced'
    )
    lr_final.fit(scaler.transform(X_tr), y_tr)
    selected_idx = np.where(np.abs(lr_final.coef_[0]) > 1e-8)[0]

    np.save("data/processed/selected_idx.npy", selected_idx)
    import pickle
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("data/processed/feature_names.pkl", "wb") as f:
        pickle.dump(feat_cols, f)

    print(f"\n[feature_select] Saved {len(selected_idx)} selected indices → "
          f"data/processed/selected_idx.npy")

    # Stability sweep
    X_all = np.vstack([X_tr, X_te])
    y_all = np.hstack([y_tr, y_te])
    stab = stability_sweep(X_all, y_all, feat_cols)
    pd.DataFrame(stab).to_csv("data/processed/stability_results.csv", index=False)

    # Print summary table
    print("\n── Feature Selection Summary ──")
    print(df_res.to_string(index=False))

    return selected_idx, feat_cols, scaler


if __name__ == "__main__":
    os.chdir("/home/claude/sparserank")
    main()
