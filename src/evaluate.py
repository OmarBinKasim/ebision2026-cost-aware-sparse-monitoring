"""
evaluate.py
-----------
Produces all paper figures and the final results tables.

Figures:
  Fig 1 — F1 vs #features trade-off curve (from feature_select_results.csv)
  Fig 2 — Anomaly score vs KPI drop scatter (with per-window coloring)
  Fig 3 — SHAP summary bar chart (top-20 sparse features)
  Fig 4 — Sparse vs full SHAP rank scatter (Spearman ρ)

Tables:
  Table 1 — Baseline comparison (full IF, sparse IF, random 20%, top-20-freq)
  Table 2 — Main results (k, F1, Precision, Recall, ROC-AUC, cost reduction)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_dataset, train_test_split_df

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = ["#2C7BB6", "#D7191C", "#1A9641", "#FDAE61", "#756BB1"]


def fig1_f1_vs_features(ax, df_sweep):
    """F1 vs #features trade-off (L1 sweep + full baseline)."""
    sparse = df_sweep[df_sweep["C"] != "FULL"].copy()
    full   = df_sweep[df_sweep["C"] == "FULL"].iloc[0]
    sparse["k_num"] = sparse["k"].astype(int)
    sparse = sparse.sort_values("k_num")

    ax.hlines(full["F1"], xmin=0, xmax=full["k"]+10,
              colors="gray", linestyles="--", linewidth=1.2, label="Full-feature IF")
    ax.hlines(full["F1"] * 0.95, xmin=0, xmax=full["k"]+10,
              colors="gray", linestyles=":", linewidth=1.0, label="95% threshold", alpha=0.7)

    ax.plot(sparse["k_num"], sparse["F1"],
            "o-", color=PALETTE[0], linewidth=2, markersize=7, label="SparseRank (L1-IF)")

    # Annotate the C values
    for _, row in sparse.iterrows():
        ax.annotate(f"C={row['C']}", xy=(row["k_num"], row["F1"]),
                    xytext=(5, 6), textcoords="offset points", fontsize=8.5, color=PALETTE[0])

    # Highlight the ≥95% knee
    passing = sparse[sparse["F1"] >= full["F1"] * 0.95]
    if len(passing):
        knee = passing.sort_values("k_num").iloc[0]
        ax.axvline(knee["k_num"], color=PALETTE[1], linewidth=1.2, linestyle="--", alpha=0.8)
        ax.scatter([knee["k_num"]], [knee["F1"]], s=100, color=PALETTE[1],
                   zorder=5, label=f"Selected k={int(knee['k_num'])}")

    ax.set_xlabel("Number of Selected Features (k)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Fig. 1: F1–Feature Count Trade-off (L1 Sweep)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, full["k"] + 20)
    ax.set_ylim(max(0, sparse["F1"].min() - 0.05), 1.01)


def fig2_score_vs_kpi(ax, scores, kpi, y_true):
    """Anomaly score vs KPI drop, colored by ground truth."""
    colors = [PALETTE[1] if l == 1 else PALETTE[2] for l in y_true]
    kpi_drop = 1.0 - kpi
    ax.scatter(scores, kpi_drop, c=colors, alpha=0.45, s=14, edgecolors="none")
    rho, pval = spearmanr(-scores, kpi_drop)
    ax.set_xlabel("IF Anomaly Score (lower = more anomalous)")
    ax.set_ylabel("KPI Drop (1 − order_http_2xx_rate)")
    ax.set_title(f"Fig. 2: Anomaly Score vs Revenue KPI Drop  (ρ={rho:.3f})")
    # Legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[1], markersize=8, label="Anomaly"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[2], markersize=8, label="Normal"),
    ]
    ax.legend(handles=handles, fontsize=9)


def fig3_shap_bar(ax, imp_df, k):
    """Top-20 SHAP feature importances (sparse)."""
    top20 = imp_df.head(20).copy()
    col = "shap_mean_abs_sparse"
    # Shorten feature names
    top20["short_name"] = top20["feature"].str.replace("__", "\n", 1)
    # Color by service
    services_in_top = top20["feature"].str.split("__").str[0]
    unique_svcs = services_in_top.unique()
    svc_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique_svcs)}
    bar_colors = [svc_color[s] for s in services_in_top]

    bars = ax.barh(range(len(top20)), top20[col].values[::-1],
                   color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["short_name"].values[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title(f"Fig. 3: Top-20 SHAP Feature Importances (k={k} sparse set)")
    ax.tick_params(axis='y', length=0)


def fig4_rank_scatter(ax, rank_df):
    """Sparse vs full SHAP rank scatter."""
    rho, _ = spearmanr(rank_df["rank_sparse"], rank_df["rank_full"])
    ax.scatter(rank_df["rank_sparse"], rank_df["rank_full"],
               alpha=0.5, s=20, color=PALETTE[0])
    mn, mx = 1, max(rank_df["rank_sparse"].max(), rank_df["rank_full"].max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.0, alpha=0.5, label="y=x")
    ax.set_xlabel("Sparse SHAP Rank")
    ax.set_ylabel("Full-model SHAP Rank")
    ax.set_title(f"Fig. 4: SHAP Ranking Correlation  (ρ={rho:.3f})")
    ax.legend(fontsize=9)


def build_baseline_table(X_tr, X_te, y_tr, y_te, selected_idx, scaler):
    """Compare SparseRank against 3 baselines."""
    from sklearn.feature_selection import SelectKBest, f_classif

    results = []
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    k = len(selected_idx)
    p = X_tr.shape[1]
    contamination = min(float(y_tr.mean()), 0.49)

    configs = [
        ("Full-feature IF",   np.arange(p),           "all features"),
        ("SparseRank (L1)",   selected_idx,            f"L1 k={k}"),
        ("Random-k IF",       None,                    f"random k={k}"),
        ("Top-k ANOVA IF",    None,                    f"ANOVA k={k}"),
        ("Top-20 freq IF",    None,                    "fixed 20 features"),
    ]

    rng = np.random.default_rng(99)

    for name, idx, note in configs:
        if name == "Random-k IF":
            idx = rng.choice(p, k, replace=False)
        elif name == "Top-k ANOVA IF":
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_tr_sc, y_tr)
            idx = selector.get_support(indices=True)
        elif name == "Top-20 freq IF":
            idx = np.arange(min(20, p))

        if_ = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        if_.fit(X_tr_sc[:, idx])
        scores = if_.decision_function(X_te_sc[:, idx])
        y_pred = (scores < 0).astype(int)
        results.append({
            "Method": name,
            "k_features": len(idx),
            "Cost_Reduction_%": round((1 - len(idx)/p)*100, 1),
            "F1": round(f1_score(y_te, y_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_te, -scores), 4),
            "Note": note
        })
        print(f"  {name:25s}  k={len(idx):3d}  F1={results[-1]['F1']:.4f}  "
              f"AUC={results[-1]['ROC_AUC']:.4f}")

    return pd.DataFrame(results)


def main():
    print("[evaluate] Loading all artifacts...")
    df = load_dataset()
    X_tr, X_te, y_tr, y_te, kpi_tr, kpi_te, feat_cols = train_test_split_df(df)

    selected_idx = np.load("data/processed/selected_idx.npy")
    scores       = np.load("data/processed/anomaly_scores.npy")
    y_pred       = np.load("data/processed/anomaly_preds.npy")
    y_te_saved   = np.load("data/processed/y_test.npy")
    kpi_te_saved = np.load("data/processed/kpi_test.npy")

    with open("data/processed/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df_sweep = pd.read_csv("data/processed/feature_select_results.csv")
    imp_df   = pd.read_csv("data/processed/shap_feature_importance.csv")
    rank_df  = pd.read_csv("data/processed/shap_rank_comparison.csv")

    # ── Build baseline comparison table ──────────────────────────────────────
    print("\n[evaluate] Building baseline comparison table...")
    baseline_df = build_baseline_table(X_tr, X_te, y_tr, y_te, selected_idx, scaler)
    baseline_df.to_csv("data/processed/baseline_comparison.csv", index=False)
    print("\n── Baseline Comparison ──")
    print(baseline_df[["Method","k_features","Cost_Reduction_%","F1","Precision","Recall","ROC_AUC"]].to_string(index=False))

    # ── Produce figures ───────────────────────────────────────────────────────
    print("\n[evaluate] Generating figures...")
    os.makedirs("experiments/figures", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SparseRank: Explainable Cost-Aware Feature Selection\nfor E-Commerce Microservice Anomaly Detection",
                 fontsize=13, y=1.01, fontweight='bold')

    fig1_f1_vs_features(axes[0, 0], df_sweep)
    fig2_score_vs_kpi(axes[0, 1], scores, kpi_te_saved, y_te_saved)
    fig3_shap_bar(axes[1, 0], imp_df, len(selected_idx))
    fig4_rank_scatter(axes[1, 1], rank_df)

    plt.tight_layout()
    plt.savefig("experiments/figures/sparserank_all_figures.pdf",
                bbox_inches="tight", dpi=150)
    plt.savefig("experiments/figures/sparserank_all_figures.png",
                bbox_inches="tight", dpi=150)
    print("  Saved: experiments/figures/sparserank_all_figures.{pdf,png}")

    # Save individual figures too
    for fig_fn, fn_stem in [
        (fig1_f1_vs_features, "fig1_f1_features"),
        (fig2_score_vs_kpi, "fig2_score_kpi"),
    ]:
        fig_single, ax_single = plt.subplots(figsize=(7, 4.5))
        if fn_stem == "fig1_f1_features":
            fig1_f1_vs_features(ax_single, df_sweep)
        elif fn_stem == "fig2_score_kpi":
            fig2_score_vs_kpi(ax_single, scores, kpi_te_saved, y_te_saved)
        plt.tight_layout()
        plt.savefig(f"experiments/figures/{fn_stem}.png", bbox_inches="tight", dpi=150)
        plt.close()

    # SHAP bar separately (needs more height)
    fig_s, ax_s = plt.subplots(figsize=(9, 7))
    fig3_shap_bar(ax_s, imp_df, len(selected_idx))
    plt.tight_layout()
    plt.savefig("experiments/figures/fig3_shap_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Rank scatter
    fig_r, ax_r = plt.subplots(figsize=(6, 5))
    fig4_rank_scatter(ax_r, rank_df)
    plt.tight_layout()
    plt.savefig("experiments/figures/fig4_rank_scatter.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("[evaluate] ✓ All figures saved.")

    # ── Pipeline timing ───────────────────────────────────────────────────────
    import time
    t0 = time.time()
    # Re-run IF prediction as a timing proxy
    with open("data/processed/if_model.pkl", "rb") as f:
        if_m = pickle.load(f)
    _ = if_m.decision_function(scaler.transform(X_te)[:, selected_idx])
    t_inf = time.time() - t0
    print(f"\n[evaluate] Inference time on {len(X_te)} windows: {t_inf*1000:.1f} ms")

    # Print final summary
    eval_metrics = pd.read_csv("data/processed/eval_metrics.csv")
    shap_summary = pd.read_csv("data/processed/shap_summary.csv")
    print("\n──────── FINAL RESULTS SUMMARY ────────")
    print(eval_metrics.T.to_string(header=False))
    print(f"\nSHAP attribution coverage: {shap_summary['attribution_coverage_pct'].iloc[0]:.1f}%")
    print(f"SHAP rank Spearman ρ:       {shap_summary['spearman_rho_rank'].iloc[0]:.4f}")


if __name__ == "__main__":
    os.chdir("/home/claude/sparserank")
    main()
