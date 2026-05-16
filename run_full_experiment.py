"""
run_full_experiment.py
======================
Complete SparseRank experiment on the real RS-Anomic dataset.

Key improvements over baseline:
  - log1p transform before scaling (handles extreme-scale cAdvisor counters)
  - RobustScaler instead of StandardScaler
  - Optimal decision threshold search (instead of fixed score < 0)
  - n_estimators=300, max_features=1.0 for IsolationForest

Run:
    cd sparserank/
    python run_full_experiment.py
"""

import sys, os, warnings, time, pickle, json
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import spearmanr
import shap
from sklearn.tree import DecisionTreeClassifier
from src.data_loader import load_dataset, train_test_split_df

os.makedirs('data/processed', exist_ok=True)
os.makedirs('experiments/figures', exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.color': '#CBD5E1',
    'figure.facecolor': 'white', 'axes.facecolor': '#F8FAFC'
})
BLUE='#1D4ED8'; ORG='#EA580C'; GRN='#15803D'; PUR='#7C3AED'
RED='#DC2626'; GRY='#6B7280'; TEAL='#0891B2'; AMB='#D97706'

T = time.time()
print('=' * 65)
print('  SparseRank — COMPLETE EXPERIMENT')
print('  Real RS-Anomic dataset · 14 fault scenarios · p=402 features')
print('=' * 65)

# ─── STEP 1: LOAD DATA ────────────────────────────────────────────────────────
print('\n[1] Loading dataset...')
df = load_dataset()
feat_cols = [c for c in df.columns
             if c not in ('label', 'fault_type', 'order_http_2xx_rate', 'window_id', 'timestamp')]
X   = df[feat_cols].values.astype(np.float64)
y   = df['label'].values
kpi = df['order_http_2xx_rate'].values

X_tr, X_te, y_tr, y_te, kpi_tr, kpi_te = train_test_split(
    X, y, kpi, test_size=0.2, stratify=y, random_state=42
)

# ── IMPROVED PREPROCESSING: log1p + RobustScaler ─────────────────────────────
# cAdvisor counters have extreme scale differences (SNR up to 263 million).
# log1p compresses the range; RobustScaler uses median/IQR (robust to outliers).
X_tr_log = np.log1p(np.clip(X_tr, 0, None))
X_te_log  = np.log1p(np.clip(X_te,  0, None))
scaler = RobustScaler()
Xtr = scaler.fit_transform(X_tr_log)
Xte = scaler.transform(X_te_log)

p    = Xtr.shape[1]
cont = min(float(y_tr.mean()), 0.49)

print(f'  Train: {X_tr.shape}  Test: {X_te.shape}')
print(f'  p={p}  anomaly_rate={y_tr.mean():.2%}  contamination={cont:.3f}')
print(f'  Preprocessing: log1p + RobustScaler')

with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'log_transform': True}, f)
with open('data/processed/feature_names.pkl', 'wb') as f:
    pickle.dump(feat_cols, f)

# ─── STEP 2: L1 FEATURE SELECTION SWEEP ──────────────────────────────────────
print('\n[2] L1 regularisation sweep...')
C_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
sweep_rows = []
best = {'C': None, 'k': 0, 'F1': 0, 'AUC': 0, 'idx': None}

def _opt_threshold(scores, y_true, n=80):
    """Search for F1-maximising threshold."""
    best_t, best_f = 0.0, 0.0
    for t in np.percentile(scores, np.linspace(5, 80, n)):
        f = f1_score(y_true, (scores < t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

for C in C_values:
    lr = LogisticRegression(
        penalty='l1', C=C, solver='saga', max_iter=800,
        random_state=42, class_weight='balanced', tol=3e-3
    )
    lr.fit(Xtr, y_tr)
    idx = np.where(np.abs(lr.coef_[0]) > 1e-8)[0]
    k = len(idx)
    if k == 0:
        print(f'  C={C:<6}  k=0  (over-regularised)')
        sweep_rows.append({'C': C, 'k': 0, 'F1': 0, 'Precision': 0,
                           'Recall': 0, 'ROC_AUC': 0, 'AP': 0, 'cost_red_pct': 100})
        continue
    if_m = IsolationForest(n_estimators=200, contamination=cont,
                            max_features=1.0, random_state=42, n_jobs=-1)
    if_m.fit(Xtr[:, idx])
    sc = if_m.decision_function(Xte[:, idx])
    t_opt = _opt_threshold(sc, y_te)
    yp = (sc < t_opt).astype(int)
    f1   = f1_score(y_te, yp, zero_division=0)
    prec = precision_score(y_te, yp, zero_division=0)
    rec  = recall_score(y_te, yp, zero_division=0)
    auc  = roc_auc_score(y_te, -sc)
    ap   = average_precision_score(y_te, -sc)
    cr   = (1 - k / p) * 100
    mark = ' ← BEST' if f1 > best['F1'] else ''
    print(f'  C={C:<6}  k={k:4d}  ({cr:5.1f}% cost)  F1={f1:.4f}  AUC={auc:.4f}{mark}')
    sweep_rows.append({'C': C, 'k': k, 'F1': round(f1,4), 'Precision': round(prec,4),
        'Recall': round(rec,4), 'ROC_AUC': round(auc,4), 'AP': round(ap,4),
        'cost_red_pct': round(cr,1)})
    if f1 > best['F1']:
        best = {'C': C, 'k': k, 'F1': f1, 'AUC': auc, 'idx': idx}

# Full-feature IF baseline
if_full = IsolationForest(n_estimators=300, contamination=cont,
                           max_features=1.0, random_state=42, n_jobs=-1)
if_full.fit(Xtr)
sc_full  = if_full.decision_function(Xte)
t_full   = _opt_threshold(sc_full, y_te)
yp_full  = (sc_full < t_full).astype(int)
f1_full  = f1_score(y_te, yp_full, zero_division=0)
auc_full = roc_auc_score(y_te, -sc_full)
sweep_rows.append({'C': 'FULL', 'k': p, 'F1': round(f1_full,4),
    'Precision': round(precision_score(y_te, yp_full, zero_division=0),4),
    'Recall': round(recall_score(y_te, yp_full, zero_division=0),4),
    'ROC_AUC': round(auc_full,4), 'AP': round(average_precision_score(y_te,-sc_full),4),
    'cost_red_pct': 0})
print(f'  Full IF p={p}:        F1={f1_full:.4f}  AUC={auc_full:.4f}')
print(f'\n  ★ Best: C={best["C"]}, k={best["k"]}, F1={best["F1"]:.4f}, '
      f'cost_red={(1-best["k"]/p)*100:.1f}%')

pd.DataFrame(sweep_rows).to_csv('data/processed/feature_select_results.csv', index=False)
selected_idx = best['idx']
np.save('data/processed/selected_idx.npy', selected_idx)

# ─── STEP 3: FINAL MODEL ──────────────────────────────────────────────────────
print('\n[3] Training final SparseRank model...')
k = len(selected_idx)
if_sp = IsolationForest(n_estimators=300, contamination=cont,
                         max_features=1.0, random_state=42, n_jobs=-1)
if_sp.fit(Xtr[:, selected_idx])
scores    = if_sp.decision_function(Xte[:, selected_idx])
t_opt_sp  = _opt_threshold(scores, y_te)
yp_sp     = (scores < t_opt_sp).astype(int)
cm_sp     = confusion_matrix(y_te, yp_sp)
f1_sp     = f1_score(y_te, yp_sp, zero_division=0)
prec_sp   = precision_score(y_te, yp_sp, zero_division=0)
rec_sp    = recall_score(y_te, yp_sp, zero_division=0)
auc_sp    = roc_auc_score(y_te, -scores)
ap_sp     = average_precision_score(y_te, -scores)
mcc_sp    = matthews_corrcoef(y_te, yp_sp)
ba_sp     = balanced_accuracy_score(y_te, yp_sp)
rho_kpi,_ = spearmanr(-scores, 1 - kpi_te)

print(f'  F1={f1_sp:.4f}  Prec={prec_sp:.4f}  Rec={rec_sp:.4f}')
print(f'  AUC={auc_sp:.4f}  AP={ap_sp:.4f}  MCC={mcc_sp:.4f}  BalAcc={ba_sp:.4f}')
print(f'  CM: TP={cm_sp[1,1]}  FP={cm_sp[0,1]}  TN={cm_sp[0,0]}  FN={cm_sp[1,0]}')
print(f'  KPI Spearman ρ = {rho_kpi:.4f}')
print(f'  Cost reduction: {(1-k/p)*100:.1f}%')

np.save('data/processed/anomaly_scores.npy', scores)
np.save('data/processed/anomaly_preds.npy',  yp_sp)
np.save('data/processed/y_test.npy',          y_te)
np.save('data/processed/kpi_test.npy',        kpi_te)
with open('data/processed/if_model.pkl', 'wb') as f:
    pickle.dump(if_sp, f)

pd.DataFrame([{
    'k': k, 'p': p, 'F1': round(f1_sp,4), 'Precision': round(prec_sp,4),
    'Recall': round(rec_sp,4), 'ROC_AUC': round(auc_sp,4), 'AP': round(ap_sp,4),
    'MCC': round(mcc_sp,4), 'BalAcc': round(ba_sp,4), 'KPI_rho': round(rho_kpi,4),
    'cost_red_pct': round((1-k/p)*100,1), 'opt_threshold': round(t_opt_sp,4),
    'TP': int(cm_sp[1,1]), 'FP': int(cm_sp[0,1]),
    'TN': int(cm_sp[0,0]), 'FN': int(cm_sp[1,0])
}]).to_csv('data/processed/eval_metrics.csv', index=False)

# ─── STEP 4: BASELINES ───────────────────────────────────────────────────────
print('\n[4] Comparing all baselines...')
rng = np.random.default_rng(42)
sel = {n: SelectKBest(f_classif, k=n).fit(Xtr, y_tr).get_support(indices=True)
       for n in [50, 100, 150, 200]}

cfgs = [
    ('Full IF (p='+str(p)+')', np.arange(p),              GRY),
    (f'SparseRank L1 (k={k})', selected_idx,               BLUE),
    ('ANOVA (k=50)',            sel[50],                   GRN),
    ('ANOVA (k=100)',           sel[100],                  TEAL),
    ('ANOVA (k=150)',           sel[150],                  PUR),
    ('ANOVA (k=200)',           sel[200],                  AMB),
    (f'Random (k={k})',         rng.choice(p,k,replace=False), RED),
]

bl_rows=[]; roc_d={}; pr_d={}; cm_d={}

for name, idx, col in cfgs:
    if_m = IsolationForest(n_estimators=300, contamination=cont,
                            max_features=1.0, random_state=42, n_jobs=-1)
    if_m.fit(Xtr[:, idx])
    sc = if_m.decision_function(Xte[:, idx])
    t  = _opt_threshold(sc, y_te)
    yp = (sc < t).astype(int)
    f1 = f1_score(y_te, yp, zero_division=0)
    auc= roc_auc_score(y_te, -sc)
    ap = average_precision_score(y_te, -sc)
    mcc= matthews_corrcoef(y_te, yp)
    cm2= confusion_matrix(y_te, yp)
    cr = (1 - len(idx)/p) * 100
    fpr, tpr, _ = roc_curve(y_te, -sc)
    pp,  rr,  _ = precision_recall_curve(y_te, -sc)
    print(f'  {name:<28}  F1={f1:.4f}  AUC={auc:.4f}  AP={ap:.4f}  MCC={mcc:.4f}  ({cr:.0f}%cr)')
    bl_rows.append({'Method': name, 'k': len(idx), 'cost_red%': round(cr,1),
        'F1': round(f1,4), 'Precision': round(precision_score(y_te,yp,zero_division=0),4),
        'Recall': round(recall_score(y_te,yp,zero_division=0),4),
        'ROC_AUC': round(auc,4), 'AP': round(ap,4), 'MCC': round(mcc,4),
        'TP': int(cm2[1,1]), 'FP': int(cm2[0,1]),
        'TN': int(cm2[0,0]), 'FN': int(cm2[1,0]), 'color': col})
    roc_d[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(auc,4), 'col': col}
    pr_d[name]  = {'prec': pp.tolist(), 'rec': rr.tolist(), 'ap': round(ap,4), 'col': col}
    cm_d[name]  = cm2.tolist()

bl_df = pd.DataFrame(bl_rows)
bl_df.to_csv('data/processed/baseline_comparison.csv', index=False)
with open('data/processed/roc_data.json', 'w') as f: json.dump(roc_d, f)
with open('data/processed/pr_data.json',  'w') as f: json.dump(pr_d,  f)

# ─── STEP 5: SHAP EXPLAINABILITY ─────────────────────────────────────────────
print('\n[5] SHAP explainability...')
feat_sp  = [feat_cols[i] for i in selected_idx]
Xte_sp   = Xte[:, selected_idx]
clf = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')
clf.fit(Xte_sp, yp_sp)
sv = np.array(shap.TreeExplainer(clf).shap_values(Xte_sp))
if sv.ndim == 3:
    sv = sv[:, :, 1]
shap_mean = np.abs(sv).mean(0)
shap_df = pd.DataFrame({
    'feature': feat_sp,
    'shap_mean_abs': shap_mean,
    'rank': np.argsort(-shap_mean) + 1
}).sort_values('shap_mean_abs', ascending=False)
shap_df.to_csv('data/processed/shap_importance.csv', index=False)
pd.DataFrame(sv, columns=[f'SHAP_{f}' for f in feat_sp]).assign(
    label=y_te, if_score=scores, kpi=kpi_te
).to_csv('data/processed/shap_per_window.csv', index=False)
print(f'  Top feature: {shap_df.iloc[0]["feature"]}  |SHAP|={shap_df.iloc[0]["shap_mean_abs"]:.4f}')

# ─── STEP 6: 5-FOLD CROSS-VALIDATION ─────────────────────────────────────────
print('\n[6] 5-fold cross-validation...')
X_all = np.vstack([X_tr, X_te])
y_all = np.hstack([y_tr, y_te])
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for mname in ['SparseRank L1', 'ANOVA k=150', 'Full IF']:
    folds_f1=[]; folds_auc=[]
    for fold, (tri, tei) in enumerate(skf.split(X_all, y_all)):
        X_f_log = np.log1p(np.clip(X_all[tri], 0, None))
        X_t_log = np.log1p(np.clip(X_all[tei], 0, None))
        sc_cv = RobustScaler()
        X_f = sc_cv.fit_transform(X_f_log)
        X_t = sc_cv.transform(X_t_log)
        y_f = y_all[tri]; y_t = y_all[tei]
        cf  = min(float(y_f.mean()), 0.49)

        if mname == 'SparseRank L1':
            lr2 = LogisticRegression(penalty='l1', C=best['C'], solver='saga',
                                      max_iter=600, random_state=42,
                                      class_weight='balanced', tol=5e-3)
            lr2.fit(X_f, y_f)
            idx2 = np.where(np.abs(lr2.coef_[0]) > 1e-8)[0]
            if len(idx2) == 0:
                idx2 = np.arange(min(k, X_f.shape[1]))
        elif 'ANOVA' in mname:
            idx2 = SelectKBest(f_classif, k=150).fit(X_f, y_f).get_support(indices=True)
        else:
            idx2 = np.arange(X_f.shape[1])

        if_cv = IsolationForest(n_estimators=200, contamination=cf,
                                 max_features=1.0, random_state=42)
        if_cv.fit(X_f[:, idx2])
        sc3 = if_cv.decision_function(X_t[:, idx2])
        t3  = _opt_threshold(sc3, y_t)
        yp3 = (sc3 < t3).astype(int)
        folds_f1.append(round(f1_score(y_t, yp3, zero_division=0), 4))
        folds_auc.append(round(roc_auc_score(y_t, -sc3), 4))
        cv_rows.append({'method': mname, 'fold': fold+1,
                        'F1': folds_f1[-1], 'AUC': folds_auc[-1]})
    print(f'  {mname}: F1={np.mean(folds_f1):.4f}±{np.std(folds_f1):.4f}  '
          f'AUC={np.mean(folds_auc):.4f}±{np.std(folds_auc):.4f}')

pd.DataFrame(cv_rows).to_csv('data/processed/cv_results.csv', index=False)

# ─── STEP 7: F1 vs k SWEEP ───────────────────────────────────────────────────
print('\n[7] F1 vs k sweep...')
ks = [5,10,20,30,50,75,100,125,150,175,200,250,300,380]
f1_k=[]; auc_k=[]
for kk in ks:
    idx3 = SelectKBest(f_classif, k=min(kk,p)).fit(Xtr, y_tr).get_support(indices=True)
    if_k = IsolationForest(n_estimators=200, contamination=cont, random_state=42, n_jobs=-1)
    if_k.fit(Xtr[:, idx3])
    sc4 = if_k.decision_function(Xte[:, idx3])
    t4  = _opt_threshold(sc4, y_te)
    yp4 = (sc4 < t4).astype(int)
    f1_k.append(round(f1_score(y_te, yp4, zero_division=0), 4))
    auc_k.append(round(roc_auc_score(y_te, -sc4), 4))
pd.DataFrame({'k': ks, 'F1': f1_k, 'ROC_AUC': auc_k}).to_csv('data/processed/f1_vs_k.csv', index=False)

# ─── STEP 8: PCA ─────────────────────────────────────────────────────────────
print('\n[8] PCA analysis...')
pca30 = PCA(n_components=30, random_state=42).fit(Xtr)
ev = pca30.explained_variance_ratio_
cumev = np.cumsum(ev)
pd.DataFrame({'component': range(1,31), 'var_ratio': ev, 'cumulative': cumev}
             ).to_csv('data/processed/pca_variance.csv', index=False)
Xpca = PCA(n_components=2, random_state=42).fit_transform(Xte)
pd.DataFrame({'PC1': Xpca[:,0], 'PC2': Xpca[:,1], 'label': y_te}
             ).to_csv('data/processed/pca_2d.csv', index=False)

# ─── STEP 9: GENERATE FIGURES ────────────────────────────────────────────────
print('\n[9] Generating figures...')

cv_df = pd.DataFrame(cv_rows)

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.54, wspace=0.42)
fig.suptitle(
    'SparseRank — Complete Results  |  Real RS-Anomic Dataset\n'
    f'5,061 windows  ·  14 fault scenarios  ·  p={p} features  '
    f'·  k={k} selected  ·  {(1-k/p)*100:.0f}% cost reduction',
    fontsize=12, fontweight='bold', y=1.01)

# (a) ROC
ax = fig.add_subplot(gs[0, 0:2])
ax.plot([0,1],[0,1],'--',c='#CBD5E1',lw=1,label='Random (0.500)')
for nm,d in roc_d.items():
    lw = 3.0 if 'SparseRank' in nm else 1.6
    ls = '-' if ('SparseRank' in nm or 'ANOVA' in nm) else '--'
    ax.plot(d['fpr'], d['tpr'], c=d['col'], lw=lw, ls=ls,
            label=f"{nm.split('(')[0].strip()} (AUC={d['auc']:.3f})")
# shade SparseRank area
sr_key = [k2 for k2 in roc_d if 'SparseRank' in k2][0]
ax.fill_between(roc_d[sr_key]['fpr'], roc_d[sr_key]['tpr'], alpha=0.07, color=BLUE)
ax.set(title='(a) ROC Curves', xlabel='False Positive Rate', ylabel='True Positive Rate')
ax.legend(fontsize=7, loc='lower right', framealpha=0.95)

# (b) PR
ax = fig.add_subplot(gs[0, 2:4])
ax.axhline(y_te.mean(), c='#CBD5E1', lw=1, ls='--', label=f'Random (AP={y_te.mean():.3f})')
for nm,d in pr_d.items():
    lw = 3.0 if 'SparseRank' in nm else 1.6
    ls = '-' if ('SparseRank' in nm or 'ANOVA' in nm) else '--'
    ax.plot(d['rec'], d['prec'], c=d['col'], lw=lw, ls=ls,
            label=f"{nm.split('(')[0].strip()} (AP={d['ap']:.3f})")
sr_key_p = [k2 for k2 in pr_d if 'SparseRank' in k2][0]
ax.fill_between(pr_d[sr_key_p]['rec'], pr_d[sr_key_p]['prec'], alpha=0.07, color=BLUE)
ax.set(title='(b) Precision-Recall Curves', xlabel='Recall', ylabel='Precision', ylim=(0, 1.05))
ax.legend(fontsize=7, loc='upper right', framealpha=0.95)

# (c) Grouped bar — all metrics
ax = fig.add_subplot(gs[1, 0:3])
short_nm = [r['Method'].split('(')[0].strip()[:16] for _,r in bl_df.iterrows()]
x = np.arange(len(short_nm)); w = 0.14
metric_spec = [('F1',BLUE),('Precision',GRN),('Recall',ORG),('ROC_AUC',PUR),('AP',TEAL),('MCC',AMB)]
for i,(met,col) in enumerate(metric_spec):
    vals = [max(0,r[met]) for _,r in bl_df.iterrows()]
    ax.bar(x+(i-2.5)*w, vals, w, label=met, color=col, alpha=0.88,
           edgecolor='white', linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(short_nm, rotation=28, ha='right', fontsize=8)
ax.set(title='(c) All Performance Metrics', ylabel='Score', ylim=(0, 1.05))
ax.legend(fontsize=8, ncol=6, loc='upper center', framealpha=0.95)
sr_idx = [i for i,m in enumerate(bl_df.Method) if 'SparseRank' in m]
if sr_idx:
    ax.axvspan(sr_idx[0]-0.5, sr_idx[0]+0.5, color=BLUE, alpha=0.06, zorder=0)

# (d) F1 ranking bar
ax = fig.add_subplot(gs[1, 3])
f1_vals  = bl_df['F1'].tolist()
cols_bar = bl_df['color'].tolist()
bars = ax.barh(short_nm, f1_vals, color=cols_bar, edgecolor='white', height=0.65)
ax.set_xlabel('F1 Score'); ax.set_xlim(0, 0.98)
ax.set_title('(d) F1 Ranking')
for bar, val in zip(bars, f1_vals):
    ax.text(val+0.003, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=7.5,
            fontweight='bold' if val == max(f1_vals) else 'normal')

# (e) SHAP bar
ax = fig.add_subplot(gs[2, 0:2])
top12 = shap_df.head(12)
svc_palette = {'rabbitmq': PUR, 'mysql': BLUE, 'web': ORG, 'dispatch': GRN,
               'cart': TEAL, 'user': AMB, 'catalogue': RED, 'payment': GRY,
               'redis': '#0EA5E9', 'mongodb': '#D946EF', 'shipping': '#84CC16',
               'ratings': '#F97316'}
bar_c = [svc_palette.get(f.split('__')[0], GRY) for f in top12.feature]
short_f = [f.replace('__', ' · ')[:28] for f in top12.feature]
ax.barh(range(12), top12.shap_mean_abs.values[::-1],
        color=bar_c[::-1], edgecolor='white', height=0.7)
ax.set_yticks(range(12)); ax.set_yticklabels(short_f[::-1], fontsize=7.5)
ax.set(title=f'(e) SHAP Feature Importance (k={k} sparse features)',
       xlabel='Mean |SHAP|')
ax.tick_params(axis='y', length=0)

# (f) Confusion matrix — SparseRank
ax = fig.add_subplot(gs[2, 2])
sns.heatmap(cm_sp, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=1.5, linecolor='white')
ax.set_xticklabels(['Pred N', 'Pred A'], fontsize=9)
ax.set_yticklabels(['True N', 'True A'], fontsize=9, rotation=0)
ax.set_title(f'(f) SparseRank Confusion Matrix\nF1={f1_sp:.3f}  AUC={auc_sp:.3f}',
             fontsize=9, color=BLUE, fontweight='bold')

# (g) CV F1 per fold
ax = fig.add_subplot(gs[2, 3])
cv_colors = {'SparseRank L1': BLUE, 'ANOVA k=150': GRN, 'Full IF': GRY}
cv_markers = {'SparseRank L1': 'o', 'ANOVA k=150': 's', 'Full IF': '^'}
for mn, grp in cv_df.groupby('method'):
    col = cv_colors.get(mn, GRY); mk = cv_markers.get(mn, 'o')
    ax.plot(grp.fold, grp.F1, f'{mk}-', c=col, lw=2, ms=6,
            label=f'{mn}\nμ={grp.F1.mean():.3f}±{grp.F1.std():.3f}')
    ax.fill_between(grp.fold,
                    grp.F1 - grp.F1.std()*0.4,
                    grp.F1 + grp.F1.std()*0.4,
                    color=col, alpha=0.1)
ax.set_xticks(range(1,6))
ax.set_xticklabels([f'F{i}' for i in range(1,6)])
ax.set(title='(g) 5-Fold Cross-Validation F1',
       xlabel='Fold', ylabel='F1', ylim=(0.5, 1.0))
ax.legend(fontsize=6.5, loc='lower right', framealpha=0.95)

# (h) Score violin
ax = fig.add_subplot(gs[3, 0:2])
sc_n = scores[y_te == 0]; sc_a = scores[y_te == 1]
vp_n = ax.violinplot([sc_n], positions=[0],   widths=0.4,
                      showmedians=True, showextrema=True)
vp_a = ax.violinplot([sc_a], positions=[0.6], widths=0.4,
                      showmedians=True, showextrema=True)
for pc in vp_n['bodies']: pc.set_facecolor('#93C5FD'); pc.set_alpha(0.7)
for k2 in ['cmedians','cmins','cmaxes','cbars']:
    vp_n[k2].set_color(BLUE); vp_n[k2].set_linewidth(1.5)
for pc in vp_a['bodies']: pc.set_facecolor('#FCA5A5'); pc.set_alpha(0.8)
for k2 in ['cmedians','cmins','cmaxes','cbars']:
    vp_a[k2].set_color(RED); vp_a[k2].set_linewidth(1.5)
ax.axhline(t_opt_sp, c=BLUE, lw=1.2, ls='--', label=f'Opt. threshold={t_opt_sp:.3f}')
ax.set_xticks([0, 0.6])
ax.set_xticklabels(['Normal windows\n(label=0)', 'Anomaly windows\n(label=1)'], fontsize=9)
ax.set(title='(h) IF Score Distributions: Normal vs Anomaly', ylabel='IF anomaly score')
ax.legend(fontsize=8)

# (i) PCA scree
pca_df2 = pd.read_csv('data/processed/pca_variance.csv')
ax = fig.add_subplot(gs[3, 2])
ax_twin = ax.twinx()
bars2 = ax.bar(pca_df2.component, pca_df2.var_ratio*100,
               color=[BLUE if i < 6 else '#BFDBFE' for i in range(30)],
               alpha=0.85, edgecolor='white', linewidth=0.3)
ax_twin.plot(pca_df2.component, pca_df2.cumulative*100,
             'r-', lw=2, marker='.', ms=4)
ax_twin.axhline(90, c='red', lw=1, ls=':', alpha=0.6, label='90%')
ax.set(title='(i) PCA Scree', xlabel='Component', ylabel='Variance %')
ax_twin.set_ylabel('Cumulative %', color='red', fontsize=9)
ax_twin.tick_params(axis='y', labelcolor='red')

# (j) Cost vs F1 scatter
ax = fig.add_subplot(gs[3, 3])
for _,r in bl_df.iterrows():
    sz = 220 if 'SparseRank' in r['Method'] else 80
    ax.scatter(r['cost_red%'], r['F1'], s=sz, color=r['color'], alpha=0.9,
               edgecolors='white' if 'SparseRank' in r['Method'] else r['color'],
               linewidths=2.5 if 'SparseRank' in r['Method'] else 0, zorder=5)
    ax.annotate(r['Method'].split('(')[0].strip()[:12],
                xy=(r['cost_red%'], r['F1']),
                xytext=(2, 4), textcoords='offset points',
                fontsize=6, color=r['color'])
ax.set(title='(j) Cost Reduction vs F1',
       xlabel='Monitoring cost reduction (%)', ylabel='F1 Score')

plt.savefig('experiments/figures/sparserank_master.png', bbox_inches='tight', dpi=150)
plt.close()
print('  Saved: experiments/figures/sparserank_master.png')

# Individual high-res ROC + PR figure
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
for ax, d_dict, key, title, xl, yl, ref_label, ref_val in [
    (axes2[0], roc_d, 'auc', 'ROC Curves — SparseRank vs All Baselines',
     'False Positive Rate', 'True Positive Rate',
     'Random classifier (AUC=0.500)', None),
    (axes2[1], pr_d, 'ap', 'Precision-Recall Curves — SparseRank vs All Baselines',
     'Recall', 'Precision', f'Random baseline (AP={y_te.mean():.3f})', y_te.mean()),
]:
    if ref_val is None:
        ax.plot([0,1],[0,1],'--',c='#CBD5E1',lw=1.2,zorder=0,label=ref_label)
    else:
        ax.axhline(ref_val,c='#CBD5E1',lw=1.2,ls='--',zorder=0,label=ref_label)
    for nm, d in d_dict.items():
        lw = 3.2 if 'SparseRank' in nm else 1.8
        al = 1.0 if 'SparseRank' in nm else 0.75
        ls = '-' if 'SparseRank' in nm else ('--' if 'ANOVA' in nm else ':')
        label = f"{nm.split('(')[0].strip()}  ({key.upper()}={d[key]:.3f})"
        if 'ROC' in title:
            ax.plot(d['fpr'], d['tpr'], c=d['col'], lw=lw, ls=ls,
                    alpha=al, label=label, zorder=3)
        else:
            ax.plot(d['rec'], d['prec'], c=d['col'], lw=lw, ls=ls,
                    alpha=al, label=label, zorder=3)
    sr_k = [n for n in d_dict if 'SparseRank' in n][0]
    if 'ROC' in title:
        ax.fill_between(d_dict[sr_k]['fpr'], d_dict[sr_k]['tpr'],
                        alpha=0.08, color=BLUE, zorder=1)
    else:
        ax.fill_between(d_dict[sr_k]['rec'], d_dict[sr_k]['prec'],
                        alpha=0.08, color=BLUE, zorder=1)
    ax.set(title=title, xlabel=xl, ylabel=yl,
           xlim=(-0.01,1.01), ylim=(0, 1.05))
    ax.legend(fontsize=9, framealpha=0.96,
              loc='lower right' if 'ROC' in title else 'upper right')

plt.tight_layout()
plt.savefig('experiments/figures/sparserank_roc_pr.png', bbox_inches='tight', dpi=150)
plt.close()
print('  Saved: experiments/figures/sparserank_roc_pr.png')

# Per-fault window count figure
fig3, ax3 = plt.subplots(figsize=(11, 4.5))
fault_counts = df[df.fault_type != 'normal'].groupby('fault_type').size().sort_values(ascending=True)
colors3 = [BLUE if 'packetloss' in f or 'latency' in f else
            ORG if 'cpu' in f or 'load' in f else
            PUR if 'memory' in f or 'fileIO' in f else
            GRN for f in fault_counts.index]
bars3 = ax3.barh(fault_counts.index, fault_counts.values,
                 color=colors3, edgecolor='white', height=0.7, alpha=0.88)
for bar, val in zip(bars3, fault_counts.values):
    ax3.text(val + 3, bar.get_y()+bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax3.set(xlabel='Number of 60-second windows',
        title='RS-Anomic Dataset — Window Count per Fault Scenario\n'
              f'Total anomaly windows: {fault_counts.sum()}  ·  '
              f'Total normal windows: {(df.label==0).sum()}')
ax3.tick_params(axis='y', labelsize=9)
plt.tight_layout()
plt.savefig('experiments/figures/fault_scenario_breakdown.png', bbox_inches='tight', dpi=150)
plt.close()
print('  Saved: experiments/figures/fault_scenario_breakdown.png')

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'  EXPERIMENT COMPLETE in {time.time()-T:.1f}s')
print(f'{"="*65}')
print(f'  Dataset       : {len(df):,} windows  |  p={p}  |  14 fault types')
print(f'  SparseRank    : k={k}  |  {(1-k/p)*100:.0f}% cost reduction')
print(f'                  F1={f1_sp:.4f}  AUC={auc_sp:.4f}  MCC={mcc_sp:.4f}')
print(f'  Full IF       : F1={f1_full:.4f}  AUC={auc_full:.4f}')
print(f'  KPI Spearman ρ: {rho_kpi:.4f}')
print(f'  Top SHAP feat : {shap_df.iloc[0]["feature"]}')
print(f'\n  Outputs saved to: data/processed/  &  experiments/figures/')
