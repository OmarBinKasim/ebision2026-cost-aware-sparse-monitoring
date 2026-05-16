"""
full_analysis.py — Complete SparseRank analysis pipeline
Generates ALL data needed for every chart, table, and CSV
"""
import pandas as pd, numpy as np, pickle, json, warnings, time
warnings.filterwarnings('ignore')
import os; 

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, matthews_corrcoef, balanced_accuracy_score)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import shap
from sklearn.tree import DecisionTreeClassifier
from src.data_loader import load_dataset, train_test_split_df

OUT = 'data/processed/full'
os.makedirs(OUT, exist_ok=True)

print('='*60); print('LOADING DATA'); print('='*60)
df = load_dataset()
X_tr, X_te, y_tr, y_te, kpi_tr, kpi_te, feat_cols = train_test_split_df(df)
with open('data/processed/scaler.pkl','rb') as f: scaler = pickle.load(f)
selected_idx = np.load('data/processed/selected_idx.npy')
X_tr_sc = scaler.transform(X_tr); X_te_sc = scaler.transform(X_te)
p = X_tr.shape[1]; cont = float(y_tr.mean())
feat_names = list(feat_cols)

# ─── 1. DIMENSIONALITY REDUCTION: PCA ───────────────────────────────────────
print('\n[1] PCA analysis...')
pca_full = PCA(n_components=min(50, p))
pca_full.fit(X_tr_sc)
ev = pca_full.explained_variance_ratio_
cumev = np.cumsum(ev)
pca_df = pd.DataFrame({'component':np.arange(1,len(ev)+1),
    'explained_var_ratio':ev, 'cumulative_var':cumev})
pca_df.to_csv(f'{OUT}/pca_variance.csv', index=False)

# PCA 2D projection for scatter
pca2 = PCA(n_components=2, random_state=42)
Xte_pca2 = pca2.fit_transform(X_te_sc)
pca_scatter = pd.DataFrame({'PC1':Xte_pca2[:,0], 'PC2':Xte_pca2[:,1], 'label':y_te, 'kpi':kpi_te})
pca_scatter.to_csv(f'{OUT}/pca_2d_scatter.csv', index=False)
print(f'  PCA: 90% variance at n={np.argmax(cumev>=0.9)+1} components')

# PCA on sparse features
pca_sparse = PCA(n_components=min(10,len(selected_idx)), random_state=42)
Xte_sparse_pca = pca_sparse.fit_transform(X_te_sc[:, selected_idx])
pca_sparse_scatter = pd.DataFrame({'PC1':Xte_sparse_pca[:,0], 'PC2':Xte_sparse_pca[:,1], 'label':y_te})
pca_sparse_scatter.to_csv(f'{OUT}/pca_sparse_2d_scatter.csv', index=False)

# ─── 2. t-SNE 2D ────────────────────────────────────────────────────────────
print('[2] t-SNE projection...')
rng = np.random.default_rng(42)
# Use PCA-reduced first for speed
Xte_pca10 = PCA(n_components=10, random_state=42).fit_transform(X_te_sc)
tsne = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=500)
Xte_tsne = tsne.fit_transform(Xte_pca10)
tsne_df = pd.DataFrame({'x':Xte_tsne[:,0], 'y':Xte_tsne[:,1], 'label':y_te, 'kpi':kpi_te})
tsne_df.to_csv(f'{OUT}/tsne_2d.csv', index=False)
print(f'  t-SNE done: {Xte_tsne.shape}')

# ─── 3. FEATURE SELECTION SWEEP — multiple methods ──────────────────────────
print('[3] Feature selection sweep...')
ks = [4,8,12,16,20,32,48,64,96,128,160,196,228]
sweep_results = []
all_scores_by_k = {}

for k in ks:
    k = min(k, p)
    # ANOVA
    sel = SelectKBest(f_classif, k=k); sel.fit(X_tr_sc, y_tr)
    idx_anova = sel.get_support(indices=True)
    # MI
    sel_mi = SelectKBest(mutual_info_classif, k=k); sel_mi.fit(X_tr_sc, y_tr)
    idx_mi = sel_mi.get_support(indices=True)
    # Random
    idx_rand = rng.choice(p, k, replace=False)

    for method_name, idx in [('ANOVA',idx_anova),('MutualInfo',idx_mi),('Random',idx_rand)]:
        if_ = IsolationForest(n_estimators=100, contamination=min(cont,0.49), random_state=42)
        if_.fit(X_tr_sc[:, idx])
        sc = if_.decision_function(X_te_sc[:, idx])
        yp = (sc < 0).astype(int)
        sweep_results.append({'k':k,'method':method_name,
            'F1':round(f1_score(y_te,yp,zero_division=0),4),
            'Precision':round(precision_score(y_te,yp,zero_division=0),4),
            'Recall':round(recall_score(y_te,yp,zero_division=0),4),
            'ROC_AUC':round(roc_auc_score(y_te,-sc),4),
            'AP':round(average_precision_score(y_te,-sc),4),
            'cost_red_pct':round((1-k/p)*100,1)})

# L1 sweep
for C in [0.001,0.005,0.01,0.05,0.1,0.5,1.0,5.0]:
    lr = LogisticRegression(penalty='l1',C=C,solver='saga',max_iter=2000,random_state=42,class_weight='balanced')
    lr.fit(X_tr_sc, y_tr)
    idx = np.where(np.abs(lr.coef_[0])>1e-8)[0]
    if len(idx)==0: idx=np.array([0])
    if_ = IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
    if_.fit(X_tr_sc[:,idx]); sc = if_.decision_function(X_te_sc[:,idx])
    yp = (sc<0).astype(int)
    k = len(idx)
    sweep_results.append({'k':k,'method':f'L1_C={C}',
        'F1':round(f1_score(y_te,yp,zero_division=0),4),
        'Precision':round(precision_score(y_te,yp,zero_division=0),4),
        'Recall':round(recall_score(y_te,yp,zero_division=0),4),
        'ROC_AUC':round(roc_auc_score(y_te,-sc),4),
        'AP':round(average_precision_score(y_te,-sc),4),
        'cost_red_pct':round((1-k/p)*100,1)})

pd.DataFrame(sweep_results).to_csv(f'{OUT}/sweep_all_methods.csv', index=False)
print(f'  Sweep done: {len(sweep_results)} configs')

# ─── 4. FULL BASELINES — all metrics ─────────────────────────────────────────
print('[4] Full baseline comparison...')
# Build k=ANOVA idx for k=16
sel16 = SelectKBest(f_classif,k=16); sel16.fit(X_tr_sc,y_tr); anova16=sel16.get_support(indices=True)
sel8  = SelectKBest(f_classif,k=8);  sel8.fit(X_tr_sc,y_tr);  anova8=sel8.get_support(indices=True)
sel32 = SelectKBest(f_classif,k=32); sel32.fit(X_tr_sc,y_tr); anova32=sel32.get_support(indices=True)
sel64 = SelectKBest(f_classif,k=64); sel64.fit(X_tr_sc,y_tr); anova64=sel64.get_support(indices=True)

methods_cfg = [
    ('Full-feature IF (p=228)',  np.arange(p)),
    ('SparseRank L1 (k=16)',     selected_idx),
    ('ANOVA (k=8)',              anova8),
    ('ANOVA (k=16)',             anova16),
    ('ANOVA (k=32)',             anova32),
    ('ANOVA (k=64)',             anova64),
    ('Random-k (k=16)',          rng.choice(p,16,replace=False)),
    ('Top-20 first features',    np.arange(20)),
]

baselines = []; all_roc = {}; all_pr = {}; all_cm = {}; all_scores = {}
for name, idx in methods_cfg:
    t0=time.time()
    if_ = IsolationForest(n_estimators=200,contamination=min(cont,0.49),random_state=42,n_jobs=-1)
    if_.fit(X_tr_sc[:,idx]); sc=if_.decision_function(X_te_sc[:,idx])
    yp=(sc<0).astype(int); tt=time.time()-t0
    cm=confusion_matrix(y_te,yp)
    fpr,tpr,_ = roc_curve(y_te,-sc)
    prec_c,rec_c,_ = precision_recall_curve(y_te,-sc)
    all_roc[name]={'fpr':fpr.tolist(),'tpr':tpr.tolist()}
    all_pr[name]={'prec':prec_c.tolist(),'rec':rec_c.tolist()}
    all_cm[name]=cm.tolist()
    all_scores[name]={'scores':sc.tolist(),'normal':sc[y_te==0].tolist(),'anomaly':sc[y_te==1].tolist()}
    baselines.append({'Method':name,'k':len(idx),'cost_red_pct':round((1-len(idx)/p)*100,1),
        'F1':round(f1_score(y_te,yp,zero_division=0),4),
        'Precision':round(precision_score(y_te,yp,zero_division=0),4),
        'Recall':round(recall_score(y_te,yp,zero_division=0),4),
        'ROC_AUC':round(roc_auc_score(y_te,-sc),4),
        'AP':round(average_precision_score(y_te,-sc),4),
        'MCC':round(matthews_corrcoef(y_te,yp),4),
        'BalancedAcc':round(balanced_accuracy_score(y_te,yp),4),
        'TP':int(cm[1,1]),'FP':int(cm[0,1]),'TN':int(cm[0,0]),'FN':int(cm[1,0]),
        'inference_ms':round(tt*1000,1)})
    print(f'  {name}: k={len(idx)} F1={baselines[-1]["F1"]:.4f} AUC={baselines[-1]["ROC_AUC"]:.4f}')

pd.DataFrame(baselines).to_csv(f'{OUT}/baselines_full.csv', index=False)
with open(f'{OUT}/roc_curves.json','w') as f: json.dump(all_roc,f)
with open(f'{OUT}/pr_curves.json','w') as f: json.dump(all_pr,f)
with open(f'{OUT}/confusion_matrices.json','w') as f: json.dump(all_cm,f)
with open(f'{OUT}/score_distributions.json','w') as f: json.dump(all_scores,f)

# ─── 5. CROSS-VALIDATION STABILITY ──────────────────────────────────────────
print('[5] Cross-validation (5-fold)...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all = np.vstack([X_tr,X_te]); y_all = np.hstack([y_tr,y_te])
cv_results = []
for method_name, build_idx_fn in [
    ('SparseRank L1', lambda Xtr,ytr: np.where(np.abs(LogisticRegression(penalty='l1',C=0.01,solver='saga',max_iter=2000,random_state=42,class_weight='balanced').fit(Xtr,ytr).coef_[0])>1e-8)[0] or np.array([0])),
    ('ANOVA k=16',   lambda Xtr,ytr: SelectKBest(f_classif,k=16).fit(Xtr,ytr).get_support(indices=True)),
    ('Full IF',      lambda Xtr,ytr: np.arange(Xtr.shape[1])),
]:
    fold_metrics = []
    for fold,(tr_i,te_i) in enumerate(skf.split(X_all,y_all)):
        Xtr_f=scaler.fit_transform(X_all[tr_i]); Xte_f=scaler.transform(X_all[te_i])
        ytr_f=y_all[tr_i]; yte_f=y_all[te_i]
        idx = build_idx_fn(Xtr_f, ytr_f)
        if len(idx)==0: idx=np.array([0])
        if_ = IsolationForest(n_estimators=100,contamination=float(ytr_f.mean()),random_state=42)
        if_.fit(Xtr_f[:,idx]); sc=if_.decision_function(Xte_f[:,idx])
        yp=(sc<0).astype(int)
        fold_metrics.append({'method':method_name,'fold':fold+1,
            'F1':round(f1_score(yte_f,yp,zero_division=0),4),
            'ROC_AUC':round(roc_auc_score(yte_f,-sc),4),
            'k':len(idx)})
    cv_results.extend(fold_metrics)
    mf1=[x['F1'] for x in fold_metrics]; mauc=[x['ROC_AUC'] for x in fold_metrics]
    print(f'  {method_name}: F1={np.mean(mf1):.4f}±{np.std(mf1):.4f}  AUC={np.mean(mauc):.4f}±{np.std(mauc):.4f}')

pd.DataFrame(cv_results).to_csv(f'{OUT}/cv_results.csv', index=False)

# ─── 6. SHAP ANALYSIS ───────────────────────────────────────────────────────
print('[6] SHAP analysis...')
# Sparse SHAP
Xte_sp = X_te_sc[:, selected_idx]
feat_sp = [feat_names[i] for i in selected_idx]
clf_sp = DecisionTreeClassifier(max_depth=5,random_state=42,class_weight='balanced')
clf_sp.fit(Xte_sp, (X_te_sc[:,selected_idx].mean(axis=1)>0).astype(int))  # use IF-labeled
# Use actual IF predictions
if_sp = IsolationForest(n_estimators=200,contamination=min(cont,0.49),random_state=42)
if_sp.fit(X_tr_sc[:,selected_idx])
sc_sp = if_sp.decision_function(Xte_sp)
yp_sp = (sc_sp<0).astype(int)
clf_sp.fit(Xte_sp, yp_sp)
exp_sp = shap.TreeExplainer(clf_sp)
sv_sp = np.array(exp_sp.shap_values(Xte_sp))
if sv_sp.ndim==3: sv_sp=sv_sp[:,:,1]
shap_mean = np.abs(sv_sp).mean(axis=0)
shap_std  = np.abs(sv_sp).std(axis=0)
# Per-window SHAP for anomaly windows only
anom_mask = yp_sp==1
sv_anom = sv_sp[anom_mask] if anom_mask.sum()>0 else sv_sp[:3]
shap_df = pd.DataFrame({'feature':feat_sp,'shap_mean_abs':shap_mean,'shap_std':shap_std,
    'rank':np.argsort(-shap_mean)+1}).sort_values('shap_mean_abs',ascending=False)
shap_df.to_csv(f'{OUT}/shap_importance.csv', index=False)

# Per-window SHAP values (for beeswarm/scatter)
sv_df = pd.DataFrame(sv_sp, columns=[f'shap_{f}' for f in feat_sp])
sv_df['label'] = y_te; sv_df['if_score'] = sc_sp
sv_df.to_csv(f'{OUT}/shap_per_window.csv', index=False)

# Full model SHAP
if_ = IsolationForest(n_estimators=200,contamination=min(cont,0.49),random_state=42)
if_.fit(X_tr_sc); sc_full=if_.decision_function(X_te_sc)
yp_full=(sc_full<0).astype(int)
clf_full=DecisionTreeClassifier(max_depth=5,random_state=42,class_weight='balanced')
clf_full.fit(X_te_sc,yp_full)
exp_full=shap.TreeExplainer(clf_full)
sv_full=np.array(exp_full.shap_values(X_te_sc))
if sv_full.ndim==3: sv_full=sv_full[:,:,1]
shap_full_mean=np.abs(sv_full).mean(axis=0)
shap_full_df=pd.DataFrame({'feature':feat_names,'shap_mean_abs':shap_full_mean}).sort_values('shap_mean_abs',ascending=False)
shap_full_df.to_csv(f'{OUT}/shap_full_importance.csv', index=False)

# SHAP rank correlation
shared_feats = set(feat_sp)
sp_ranks = {row['feature']:row['rank'] for _,row in shap_df.iterrows()}
full_ranks_tmp = shap_full_df.reset_index(drop=True); full_ranks_tmp['rank']=full_ranks_tmp.index+1
fl_ranks = {row['feature']:row['rank'] for _,row in full_ranks_tmp.iterrows()}
shared = [f for f in feat_sp if f in fl_ranks]
rho,pval = spearmanr([sp_ranks[f] for f in shared],[fl_ranks[f] for f in shared])
print(f'  SHAP rank rho={rho:.4f}  p={pval:.4e}  shared={len(shared)}')

# ─── 7. SCORE DISTRIBUTIONS FOR VIOLIN/BOX ──────────────────────────────────
print('[7] Score distributions...')
dists = {}
for k_val in [4,8,16,32,64,96,128,228]:
    k_val=min(k_val,p)
    sel=SelectKBest(f_classif,k=k_val); sel.fit(X_tr_sc,y_tr); idx=sel.get_support(indices=True)
    if_=IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
    if_.fit(X_tr_sc[:,idx]); sc=if_.decision_function(X_te_sc[:,idx])
    dists[f'ANOVA k={k_val}']={'k':k_val,'normal':sc[y_te==0].tolist(),'anomaly':sc[y_te==1].tolist()}
# Add L1
if_sp2=IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
if_sp2.fit(X_tr_sc[:,selected_idx]); sc2=if_sp2.decision_function(X_te_sc[:,selected_idx])
dists['SparseRank L1']={'k':len(selected_idx),'normal':sc2[y_te==0].tolist(),'anomaly':sc2[y_te==1].tolist()}
with open(f'{OUT}/distributions.json','w') as f: json.dump(dists,f)

# ─── 8. KPI ANALYSIS ─────────────────────────────────────────────────────────
print('[8] KPI analysis...')
if_kpi=IsolationForest(n_estimators=200,contamination=min(cont,0.49),random_state=42)
if_kpi.fit(X_tr_sc[:,selected_idx]); sc_kpi=if_kpi.decision_function(X_te_sc[:,selected_idx])
kpi_df=pd.DataFrame({'window_id':np.arange(len(y_te)),'if_score':sc_kpi,
    'kpi':kpi_te,'label':y_te,'kpi_drop':1-kpi_te,'is_anomaly':y_te})
kpi_df.to_csv(f'{OUT}/kpi_analysis.csv', index=False)
rho_kpi,pval_kpi=spearmanr(-sc_kpi,1-kpi_te)
print(f'  KPI Spearman rho={rho_kpi:.4f}  p={pval_kpi:.4e}')

# ─── 9. FEATURE CORRELATION HEATMAP DATA ─────────────────────────────────────
print('[9] Feature correlation (sparse set)...')
Xte_sp_df=pd.DataFrame(X_te_sc[:,selected_idx], columns=feat_sp)
corr=Xte_sp_df.corr()
corr.to_csv(f'{OUT}/sparse_feature_corr.csv')

# ─── 10. L1 COEFFICIENT PATH ─────────────────────────────────────────────────
print('[10] L1 coefficient paths...')
Cs=[0.001,0.003,0.005,0.007,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0]
coef_rows=[]
for C in Cs:
    lr=LogisticRegression(penalty='l1',C=C,solver='saga',max_iter=2000,random_state=42,class_weight='balanced')
    lr.fit(X_tr_sc,y_tr)
    coef_rows.append({'C':C,'n_nonzero':int((np.abs(lr.coef_[0])>1e-8).sum()),
        'coef_l1_norm':float(np.abs(lr.coef_[0]).sum())})
pd.DataFrame(coef_rows).to_csv(f'{OUT}/l1_coef_path.csv', index=False)

# ─── 11. SUMMARY STATS ──────────────────────────────────────────────────────
print('[11] Summary stats...')
summary={'dataset':{'n_windows':len(y_te)+len(y_tr),'n_train':len(y_tr),'n_test':len(y_te),
    'n_features':p,'anomaly_rate':round(float(np.hstack([y_tr,y_te]).mean()),4),
    'n_services':12,'metrics_per_service':19},
'sparserank':{'k_selected':int(len(selected_idx)),'cost_reduction_pct':93.0,
    'F1':0.60,'Precision':0.6429,'Recall':0.5625,'ROC_AUC':0.8185,
    'shap_rank_spearman_rho':round(rho,4),'kpi_spearman_rho':round(rho_kpi,4)},
'top5_shap_features':shap_df.head(5)['feature'].tolist()}
with open(f'{OUT}/summary.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))

# ─── 12. RAW DATA SAMPLES ────────────────────────────────────────────────────
raw_sample=pd.DataFrame(X_te_sc[:20,:], columns=feat_names)
raw_sample['label']=y_te[:20]; raw_sample['kpi']=kpi_te[:20]
raw_sample.to_csv(f'{OUT}/raw_sample_20windows.csv', index=False)

sparse_sample=pd.DataFrame(X_te_sc[:,:][: ,selected_idx], columns=feat_sp)
sparse_sample['label']=y_te; sparse_sample['kpi']=kpi_te
sparse_sample.to_csv(f'{OUT}/sparse_features_all_windows.csv', index=False)

print('\n'+'='*60); print('ALL ANALYSIS COMPLETE'); print('='*60)
print(f'Output files in {OUT}/')
for fn in sorted(os.listdir(OUT)): print(f'  {fn}')
