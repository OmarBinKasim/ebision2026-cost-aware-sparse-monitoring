import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/murad/Downloads/SparseRank_Final (1)/SparseRank_Final/')

import numpy as np, pandas as pd, pickle, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import shap
from sklearn.tree import DecisionTreeClassifier
from src.data_loader import load_dataset, train_test_split_df

OUT = 'experiments/figures'
os.makedirs(OUT, exist_ok=True)

df_data = load_dataset()
X_tr, X_te, y_tr, y_te, kpi_tr, kpi_te, feat_cols = train_test_split_df(df_data)
with open('data/processed/scaler.pkl','rb') as f: scaler = pickle.load(f)
selected_idx = np.load('data/processed/selected_idx.npy')
X_tr_sc = scaler.transform(X_tr); X_te_sc = scaler.transform(X_te)
p = X_tr.shape[1]; cont = float(y_tr.mean())
feat_names = list(feat_cols)
feat_sp = [feat_names[i] for i in selected_idx]

STYLE = {'figure.dpi':150,'font.family':'DejaVu Sans','font.size':10,
    'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10,
    'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'#F8F9FA',
    'axes.grid':True,'grid.alpha':0.4,'grid.color':'#D1D5DB',
    'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':9}
plt.rcParams.update(STYLE)

BLUE='#2563EB'; ORG='#EA580C'; GRN='#16A34A'; PUR='#7C3AED'
TEAL='#0891B2'; RED='#DC2626'; AMB='#D97706'; GRY='#6B7280'
PINK='#DB2777'; LIM='#65A30D'

rng = np.random.default_rng(42)
sel8  = SelectKBest(f_classif,k=8).fit(X_tr_sc,y_tr)
sel16 = SelectKBest(f_classif,k=16).fit(X_tr_sc,y_tr)
sel32 = SelectKBest(f_classif,k=32).fit(X_tr_sc,y_tr)
sel64 = SelectKBest(f_classif,k=64).fit(X_tr_sc,y_tr)
a8=sel8.get_support(indices=True); a16=sel16.get_support(indices=True)
a32=sel32.get_support(indices=True); a64=sel64.get_support(indices=True)

cfgs = [('Full IF (p=228)',np.arange(p),GRY),
        ('SparseRank L1 (k=16)',selected_idx,BLUE),
        ('ANOVA (k=8)',a8,TEAL),
        ('ANOVA (k=16)',a16,GRN),
        ('ANOVA (k=32)',a32,LIM),
        ('ANOVA (k=64)',a64,PUR),
        ('Random (k=16)',rng.choice(p,16,replace=False),RED),
        ('Top-20 freq',np.arange(20),AMB)]

results=[]; roc_data={}; pr_data={}; cm_data={}; score_data={}
for name,idx,color in cfgs:
    if_=IsolationForest(n_estimators=200,contamination=min(cont,0.49),random_state=42,n_jobs=-1)
    if_.fit(X_tr_sc[:,idx]); sc=if_.decision_function(X_te_sc[:,idx]); yp=(sc<0).astype(int)
    fpr,tpr,_=roc_curve(y_te,-sc); prec_c,rec_c,_=precision_recall_curve(y_te,-sc)
    results.append({'name':name,'k':len(idx),'color':color,
        'F1':f1_score(y_te,yp,zero_division=0),'Precision':precision_score(y_te,yp,zero_division=0),
        'Recall':recall_score(y_te,yp,zero_division=0),'AUC':roc_auc_score(y_te,-sc),
        'AP':average_precision_score(y_te,-sc),'MCC':matthews_corrcoef(y_te,yp),
        'cost_red':(1-len(idx)/p)*100,'cm':confusion_matrix(y_te,yp)})
    roc_data[name]={'fpr':fpr,'tpr':tpr,'auc':roc_auc_score(y_te,-sc),'color':color}
    pr_data[name]={'prec':prec_c,'rec':rec_c,'ap':average_precision_score(y_te,-sc),'color':color}
    score_data[name]={'n':sc[y_te==0],'a':sc[y_te==1],'color':color}
print('Baselines ready')

# ============================================================
# FIG 1: SHAP Feature Importance
# ============================================================
shap_df = pd.read_csv('data/processed/full/shap_importance.csv')
feat_short = [f.replace('frontend__','fe·').replace('order__','ord·').replace('inventory__','inv·')
    .replace('cart__','crt·').replace('product__','prd·').replace('shipping__','shp·')
    .replace('auth__','aut·').replace('gateway__','gtw·').replace('response_time_p99','resp_p99')
    .replace('db_connections_active','db_conn').replace('http_error_rate','http_err')
    .replace('cpu_usage_percent','cpu').replace('thread_pool_queue','thrd_q')
    .replace('fs_reads_bytes','fs_rd') for f in shap_df['feature']]
svc_colors = []
for f in shap_df['feature']:
    if 'frontend' in f: svc_colors.append(ORG)
    elif 'order' in f: svc_colors.append(BLUE)
    elif 'inventory' in f: svc_colors.append(GRN)
    else: svc_colors.append(PUR)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(range(len(shap_df)), shap_df['shap_mean'], color=svc_colors, edgecolor='white', linewidth=0.5, height=0.7)
ax.set_yticks(range(len(shap_df)))
ax.set_yticklabels(feat_short, fontsize=8.5)
ax.set_xlabel('Mean |SHAP| Value')
ax.set_title('SHAP Feature Importance — SparseRank Sparse Set (k=16)', pad=12)
ax.axvline(0, color='#374151', linewidth=0.8)
for i, (v, bar) in enumerate(zip(shap_df['shap_mean'], bars)):
    if v > 0.005:
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8, color='#374151')
from matplotlib.patches import Patch
legend_handles = [Patch(color=ORG, label='frontend'), Patch(color=BLUE, label='order'),
    Patch(color=GRN, label='inventory'), Patch(color=PUR, label='other')]
ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9, fontsize=8.5)
plt.tight_layout()
plt.savefig(f'{OUT}/fig01_shap_importance.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig01 saved')

# ============================================================
# FIG 2: Grouped Performance Bar Chart
# ============================================================
metrics = ['F1','Precision','Recall','AUC','AP','MCC']
met_colors = [BLUE, GRN, ORG, PUR, TEAL, AMB]
names_short = [r['name'].replace(' (p=228)','').replace(' (k=16)','').replace(' (k=8)','').replace(' (k=16)','').replace(' (k=32)','').replace(' (k=64)','') for r in results]
n_m = len(metrics); n_g = len(results)
x = np.arange(n_g); width = 0.12

fig, ax = plt.subplots(figsize=(14, 5))
for i, (met, col) in enumerate(zip(metrics, met_colors)):
    vals = [max(0, r[met if met != 'AUC' else 'AUC']) for r in results]
    bars = ax.bar(x + i*width - (n_m-1)*width/2, vals, width, label=met, color=col, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels([r['name'].replace(' (p=228)','').replace(' (k=16)','').replace(' (k=32)','').replace(' (k=64)','').replace(' (k=8)','') for r in results], fontsize=8, rotation=15, ha='right')
ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
ax.set_title('Performance Metrics Comparison — All Feature Selection Methods', pad=12)
ax.legend(loc='upper right', ncol=6, framealpha=0.9)
ax.axvline(0.5, color='#2563EB', linewidth=1.5, linestyle='--', alpha=0.4)
ax.annotate('★ SparseRank L1', xy=(1, 0.62), fontsize=9, color=BLUE, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig02_performance_metrics_bar.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig02 saved')

# ============================================================
# FIG 3: ROC Curves
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0,1],[0,1],'--', color='#9CA3AF', linewidth=1, label='Random (AUC=0.50)')
for name, d in roc_data.items():
    lw = 2.5 if 'SparseRank' in name else 1.5
    ls = '-' if 'SparseRank' in name or 'ANOVA' in name else '--'
    ax.plot(d['fpr'], d['tpr'], color=d['color'], linewidth=lw, linestyle=ls,
        label=f"{name.replace(' (p=228)','').replace(' (k=16)','').replace(' (k=8)','').replace(' (k=32)','').replace(' (k=64)','')} (AUC={d['auc']:.3f})")
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Methods', pad=12)
ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
ax.set_xlim(-0.01,1); ax.set_ylim(0,1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig03_roc_curves.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig03 saved')

# ============================================================
# FIG 4: Precision-Recall Curves
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.axhline(cont, color='#9CA3AF', linewidth=1, linestyle='--', label=f'Baseline (AP={cont:.3f})')
for name, d in pr_data.items():
    lw = 2.5 if 'SparseRank' in name else 1.5
    ls = '-' if 'SparseRank' in name or 'ANOVA' in name else '--'
    ax.plot(d['rec'], d['prec'], color=d['color'], linewidth=lw, linestyle=ls,
        label=f"{name.replace(' (p=228)','').replace(' (k=16)','').replace(' (k=8)','').replace(' (k=32)','').replace(' (k=64)','')} (AP={d['ap']:.3f})")
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — All Methods', pad=12)
ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
ax.set_xlim(0,1.01); ax.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig(f'{OUT}/fig04_pr_curves.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig04 saved')

# ============================================================
# FIG 5: Confusion Matrices (2x4 grid)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, r in zip(axes.flat, results):
    cm = r['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
        annot_kws={'size':14,'weight':'bold'}, linewidths=1, linecolor='white')
    ax.set_xticklabels(['Pred N','Pred A'], fontsize=9)
    ax.set_yticklabels(['True N','True A'], fontsize=9, rotation=0)
    sname = r['name'].replace(' (p=228)','').replace(' (k=16)','').replace(' (k=8)','').replace(' (k=32)','').replace(' (k=64)','')
    col = r['color'] if 'SparseRank' in r['name'] else 'black'
    weight = 'bold' if 'SparseRank' in r['name'] else 'normal'
    ax.set_title(f"{sname}\nF1={r['F1']:.3f}  AUC={r['AUC']:.3f}", fontsize=9, color=col, fontweight=weight)
    tp,fp,fn,tn = cm[1,1],cm[0,1],cm[1,0],cm[0,0]
fig.suptitle('Confusion Matrices — All Methods', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig05_confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig05 saved')

# ============================================================
# FIG 6: Violin Plot (score distributions)
# ============================================================
dist_cfgs = [('ANOVA k=4',a8[:4],GRY),('ANOVA k=8',a8,TEAL),('ANOVA k=16',a16,GRN),
             ('ANOVA k=32',a32,LIM),('ANOVA k=64',a64,PUR),('ANOVA k=228',np.arange(p),GRY),('SparseRank L1',selected_idx,BLUE)]
fig, ax = plt.subplots(figsize=(13, 5))
positions_n=[]; positions_a=[]; data_n=[]; data_a=[]; xlabels=[]
for i,(name,idx,col) in enumerate(dist_cfgs):
    if_d=IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
    if_d.fit(X_tr_sc[:,idx]); sc=if_d.decision_function(X_te_sc[:,idx])
    data_n.append(sc[y_te==0]); data_a.append(sc[y_te==1])
    positions_n.append(i*2.5+0.3); positions_a.append(i*2.5-0.3)
    xlabels.append(name.replace('ANOVA ','k=').replace('SparseRank L1','L1★'))

vp_n = ax.violinplot(data_n, positions=positions_n, widths=0.5, showmedians=True, showextrema=True)
vp_a = ax.violinplot(data_a, positions=positions_a, widths=0.5, showmedians=True, showextrema=True)
for pc in vp_n['bodies']: pc.set_facecolor('#93C5FD'); pc.set_alpha(0.6)
for key in ['cmedians','cmins','cmaxes','cbars']: vp_n[key].set_color(BLUE); vp_n[key].set_linewidth(1.5)
for pc in vp_a['bodies']: pc.set_facecolor('#FCA5A5'); pc.set_alpha(0.7)
for key in ['cmedians','cmins','cmaxes','cbars']: vp_a[key].set_color(RED); vp_a[key].set_linewidth(1.5)
ax.scatter(positions_n, [np.median(d) for d in data_n], color=BLUE, s=30, zorder=5)
ax.scatter(positions_a, [np.median(d) for d in data_a], color=RED, s=30, zorder=5)
ax.set_xticks([i*2.5 for i in range(len(dist_cfgs))])
ax.set_xticklabels(xlabels, fontsize=8.5, rotation=15, ha='right')
ax.set_ylabel('IF Anomaly Score'); ax.set_title('Violin Plot — IF Score Distributions: Normal vs Anomaly Windows', pad=12)
ax.axhline(0, color='#374151', linewidth=1, linestyle='--', alpha=0.5, label='Decision threshold (0)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#93C5FD',alpha=0.7,label='Normal windows'),
    Patch(color='#FCA5A5',alpha=0.7,label='Anomaly windows')], loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig06_violin_plots.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig06 saved')

# ============================================================
# FIG 7: Box Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, label, data_list, col in [(axes[0],'Normal Windows',data_n,BLUE),(axes[1],'Anomaly Windows',data_a,RED)]:
    bp = ax.boxplot(data_list, patch_artist=True, notch=False, vert=True,
        medianprops={'color':'white','linewidth':2.5},
        whiskerprops={'linewidth':1.5},flierprops={'marker':'o','markersize':4,'alpha':0.5})
    for i,(patch,name) in enumerate(zip(bp['boxes'],xlabels)):
        alpha = 0.9 if i==6 else 0.6
        fc = col if i==6 else '#D1D5DB'
        patch.set_facecolor(fc); patch.set_alpha(alpha)
        if i==6: patch.set_edgecolor(col); patch.set_linewidth(2)
    ax.set_xticklabels(xlabels, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylabel('IF Anomaly Score'); ax.set_title(f'Box Plot — {label}', pad=10)
    ax.axhline(0, color='#374151', linewidth=1, linestyle='--', alpha=0.5)
fig.suptitle('Box Plots — IF Score Distribution by Feature Class', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig07_box_plots.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig07 saved')

# ============================================================
# FIG 8: Scatter — PCA 2D
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
pca_full = PCA(n_components=2, random_state=42).fit(X_tr_sc)
Xpca = pca_full.transform(X_te_sc)
ax = axes[0]
ax.scatter(Xpca[y_te==0,0], Xpca[y_te==0,1], c=BLUE, alpha=0.3, s=15, label=f'Normal (n={y_te.sum()==0})')
ax.scatter(Xpca[y_te==1,0], Xpca[y_te==1,1], c=RED, alpha=0.85, s=60, marker='*', label=f'Anomaly (n={y_te.sum()})', zorder=5)
ax.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('PCA 2D — Full Feature Space (p=228)', pad=10)
ax.legend(fontsize=9)

pca_sp = PCA(n_components=2, random_state=42).fit(X_tr_sc[:,selected_idx])
Xpca_s = pca_sp.transform(X_te_sc[:,selected_idx])
ax = axes[1]
ax.scatter(Xpca_s[y_te==0,0], Xpca_s[y_te==0,1], c=BLUE, alpha=0.3, s=15, label='Normal')
ax.scatter(Xpca_s[y_te==1,0], Xpca_s[y_te==1,1], c=RED, alpha=0.85, s=60, marker='*', label='Anomaly', zorder=5)
ax.set_xlabel(f'PC1 ({pca_sp.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca_sp.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('PCA 2D — Sparse Features (k=16)', pad=10)
ax.legend(fontsize=9)
fig.suptitle('PCA 2D Projections — Full vs Sparse Feature Space', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig08_pca_scatter.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig08 saved')

# ============================================================
# FIG 9: t-SNE Scatter
# ============================================================
X10 = PCA(n_components=10,random_state=42).fit_transform(X_te_sc)
tsne = TSNE(n_components=2,perplexity=20,random_state=42,max_iter=500)
Xt = tsne.fit_transform(X10)
fig, ax = plt.subplots(figsize=(7, 6))
sc1 = ax.scatter(Xt[y_te==0,0], Xt[y_te==0,1], c=BLUE, alpha=0.25, s=20, label='Normal')
sc2 = ax.scatter(Xt[y_te==1,0], Xt[y_te==1,1], c=RED, alpha=0.9, s=80, marker='*', label='Anomaly', zorder=5, edgecolors='darkred', linewidths=0.5)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE 2D Projection — Full Feature Space (p=228)', pad=12)
ax.legend(fontsize=9); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(f'{OUT}/fig09_tsne_scatter.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig09 saved')

# ============================================================
# FIG 10: F1 vs k sweep (multiple selection methods)
# ============================================================
ks = [4,8,12,16,20,32,48,64,96,128,160,228]
sweep = {'ANOVA':[],'MutualInfo':[],'Random':[]}
for k in ks:
    k=min(k,p)
    for mn, mfn in [('ANOVA',f_classif),('MutualInfo',mutual_info_classif)]:
        sel=SelectKBest(mfn,k=k); sel.fit(X_tr_sc,y_tr); idx=sel.get_support(indices=True)
        if_d=IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
        if_d.fit(X_tr_sc[:,idx]); sc2=if_d.decision_function(X_te_sc[:,idx])
        sweep[mn].append(f1_score(y_te,(sc2<0).astype(int),zero_division=0))
    idx_r=rng.choice(p,k,replace=False)
    if_d=IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
    if_d.fit(X_tr_sc[:,idx_r]); sc2=if_d.decision_function(X_te_sc[:,idx_r])
    sweep['Random'].append(f1_score(y_te,(sc2<0).astype(int),zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(ks, sweep['ANOVA'], 'o-', color=GRN, linewidth=2, markersize=5, label='ANOVA (supervised)')
ax.plot(ks, sweep['MutualInfo'], 's-', color=TEAL, linewidth=2, markersize=5, linestyle='--', label='Mutual Info')
ax.plot(ks, sweep['Random'], '^-', color=GRY, linewidth=1.5, markersize=5, linestyle=':', alpha=0.7, label='Random baseline')
ax.axhline(results[0]['F1'], color=GRY, linewidth=1.5, linestyle='-.', alpha=0.7, label='Full IF baseline')
ax.scatter([16],[results[1]['F1']], color=BLUE, s=150, zorder=6, marker='*', label=f"SparseRank L1 (F1={results[1]['F1']:.3f})")
ax.axvline(16, color=BLUE, linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Features (k)'); ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs Feature Count — Selection Methods', pad=10)
ax.legend(fontsize=8.5); ax.set_ylim(0, 0.85)

ax2 = axes[1]
Cs = [0.001,0.003,0.005,0.008,0.01,0.02,0.05,0.1,0.5,1.0,5.0]
l1_ks=[]; l1_f1=[]
for C in Cs:
    lr=LogisticRegression(penalty='l1',C=C,solver='saga',max_iter=2000,random_state=42,class_weight='balanced')
    lr.fit(X_tr_sc,y_tr); idx2=np.where(np.abs(lr.coef_[0])>1e-8)[0]
    if len(idx2)==0: idx2=np.array([0])
    if_d=IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
    if_d.fit(X_tr_sc[:,idx2]); sc3=if_d.decision_function(X_te_sc[:,idx2])
    l1_ks.append(len(idx2)); l1_f1.append(f1_score(y_te,(sc3<0).astype(int),zero_division=0))

ax2_twin = ax2.twinx()
ax2.plot(Cs, l1_f1, 'o-', color=BLUE, linewidth=2.5, markersize=6, label='F1 (left)', zorder=5)
ax2_twin.plot(Cs, l1_ks, 's--', color=ORG, linewidth=2, markersize=5, label='k selected (right)', alpha=0.7)
ax2.scatter([0.01],[results[1]['F1']], color=RED, s=200, zorder=6, marker='*')
ax2.annotate('★ C=0.01\nk=16, F1=0.58', xy=(0.01, results[1]['F1']), xytext=(0.08, 0.65),
    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color=RED)
ax2.set_xscale('log'); ax2.set_xlabel('Regularization C (log scale)')
ax2.set_ylabel('F1 Score', color=BLUE); ax2_twin.set_ylabel('Features Selected (k)', color=ORG)
ax2.tick_params(axis='y', labelcolor=BLUE); ax2_twin.tick_params(axis='y', labelcolor=ORG)
ax2.set_title('L1 Regularization Path — C vs F1 and k', pad=10)
fig.suptitle('Dimensionality Reduction Analysis', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_sweep_and_l1_path.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig10 saved')

# ============================================================
# FIG 11: Scatter — Cost Reduction vs F1 (bubble)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
for r in results:
    size = 300 if 'SparseRank' in r['name'] else 150
    zord = 5 if 'SparseRank' in r['name'] else 3
    ax.scatter(r['cost_red'], r['F1'], s=size, color=r['color'], alpha=0.85, zorder=zord,
        edgecolors='white' if 'SparseRank' in r['name'] else r['color'], linewidths=2 if 'SparseRank' in r['name'] else 0)
    name_label = r['name'].replace(' (p=228)','').replace(' (k=16)','').replace(' (k=8)','').replace(' (k=32)','').replace(' (k=64)','')
    offset = (2, 0.01) if 'SparseRank' in r['name'] else (2, -0.025)
    ax.annotate(f"{name_label}\n(k={r['k']})", xy=(r['cost_red'], r['F1']),
        xytext=(r['cost_red']+offset[0], r['F1']+offset[1]), fontsize=7.5,
        color='#1E40AF' if 'SparseRank' in r['name'] else '#374151',
        fontweight='bold' if 'SparseRank' in r['name'] else 'normal')
ax.set_xlabel('Monitoring Cost Reduction (%)'); ax.set_ylabel('F1 Score')
ax.set_title('Cost Reduction vs Detection Performance — All Methods', pad=12)
ax.set_xlim(-5, 105); ax.set_ylim(-0.05, 0.85)
ax.axvline(93, color=BLUE, linewidth=1, linestyle='--', alpha=0.4)
ax.axhline(results[1]['F1'], color=BLUE, linewidth=1, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/fig11_cost_vs_f1_scatter.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig11 saved')

# ============================================================
# FIG 12: Cross-Validation
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all = np.vstack([X_tr,X_te]); y_all = np.hstack([y_tr,y_te])
scaler_cv = StandardScaler(); cv_res = {}

for method_name in ['SparseRank L1', 'ANOVA k=16', 'Full IF (p=228)']:
    fold_f1=[]; fold_auc=[]
    for tr_i,te_i in skf.split(X_all,y_all):
        Xtr_f=scaler_cv.fit_transform(X_all[tr_i]); Xte_f=scaler_cv.transform(X_all[te_i])
        ytr_f=y_all[tr_i]; yte_f=y_all[te_i]; cf=float(ytr_f.mean())
        if method_name=='SparseRank L1':
            lr=LogisticRegression(penalty='l1',C=0.01,solver='saga',max_iter=2000,random_state=42,class_weight='balanced')
            lr.fit(Xtr_f,ytr_f); idx3=np.where(np.abs(lr.coef_[0])>1e-8)[0]
            if len(idx3)==0: idx3=np.array([0])
        elif method_name=='ANOVA k=16':
            idx3=SelectKBest(f_classif,k=16).fit(Xtr_f,ytr_f).get_support(indices=True)
        else: idx3=np.arange(p)
        if_cv=IsolationForest(n_estimators=100,contamination=min(cf,0.49),random_state=42)
        if_cv.fit(Xtr_f[:,idx3]); sc4=if_cv.decision_function(Xte_f[:,idx3]); yp4=(sc4<0).astype(int)
        fold_f1.append(f1_score(yte_f,yp4,zero_division=0))
        fold_auc.append(roc_auc_score(yte_f,-sc4))
    cv_res[method_name]={'f1':fold_f1,'auc':fold_auc}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
folds = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5']
cv_colors = {'SparseRank L1':BLUE,'ANOVA k=16':GRN,'Full IF (p=228)':GRY}
markers = {'SparseRank L1':'o','ANOVA k=16':'s','Full IF (p=228)':'^'}

for met, ax, title in [('f1',axes[0],'F1 Score'),('auc',axes[1],'ROC-AUC')]:
    for mn,d in cv_res.items():
        vals = d[met]
        col = cv_colors[mn]; mk = markers[mn]
        ax.plot(folds, vals, f'{mk}-', color=col, linewidth=2, markersize=7, label=f"{mn} (μ={np.mean(vals):.3f}±{np.std(vals):.3f})")
        ax.fill_between(range(5), [v-np.std(vals) for v in vals], [v+np.std(vals) for v in vals], color=col, alpha=0.08)
    ax.set_ylabel(title); ax.set_title(f'5-Fold CV — {title}', pad=10)
    ax.legend(fontsize=8.5); ax.set_ylim(0, 1.05)
    ax.set_xticks(range(5)); ax.set_xticklabels(folds, fontsize=9)
fig.suptitle('5-Fold Cross-Validation Stability', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig12_cross_validation.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig12 saved')

# ============================================================
# FIG 13: PCA Scree + Cumulative Variance
# ============================================================
pca_all = PCA(n_components=30, random_state=42).fit(X_tr_sc)
ev = pca_all.explained_variance_ratio_
cumev = np.cumsum(ev)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
bars = ax.bar(range(1,31), ev*100, color=[BLUE if i<5 else '#93C5FD' for i in range(30)], edgecolor='white')
ax.set_xlabel('Principal Component'); ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Scree Plot — Individual Explained Variance', pad=10)
ax.set_xlim(0, 31)

ax2 = axes[1]
ax2.plot(range(1,31), cumev*100, 'o-', color=BLUE, linewidth=2.5, markersize=5)
ax2.axhline(90, color=RED, linewidth=1.5, linestyle='--', alpha=0.7, label='90% threshold')
ax2.axhline(95, color=ORG, linewidth=1.5, linestyle='--', alpha=0.7, label='95% threshold')
n90=np.argmax(cumev>=0.9)+1; n95=np.argmax(cumev>=0.95)+1
ax2.axvline(n90, color=RED, linewidth=1, linestyle=':', alpha=0.6)
ax2.axvline(n95, color=ORG, linewidth=1, linestyle=':', alpha=0.6)
ax2.annotate(f'90% @ PC{n90}', xy=(n90, 90), xytext=(n90+1, 82), fontsize=9, color=RED)
ax2.annotate(f'95% @ PC{n95}', xy=(n95, 95), xytext=(n95+1, 87), fontsize=9, color=ORG)
ax2.set_xlabel('Number of Components'); ax2.set_ylabel('Cumulative Variance (%)')
ax2.set_title('PCA Cumulative Explained Variance', pad=10)
ax2.legend(fontsize=9); ax2.set_ylim(0, 105)
fig.suptitle('Dimensionality Analysis — PCA Explained Variance', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig13_pca_variance.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig13 saved')

# ============================================================
# FIG 14: Correlation Heatmap (sparse features)
# ============================================================
Xdf = pd.DataFrame(X_te_sc[:,selected_idx], columns=[f.replace('frontend__','fe·').replace('order__','ord·').replace('inventory__','inv·').replace('cart__','crt·').replace('product__','prd·').replace('shipping__','shp·').replace('auth__','aut·').replace('gateway__','gtw·').replace('response_time_p99','resp').replace('db_connections_active','db_con').replace('http_error_rate','http_e').replace('cpu_usage_percent','cpu').replace('thread_pool_queue','thrd_q').replace('fs_reads_bytes','fs_rd') for f in feat_sp])
corr = Xdf.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
cmap = LinearSegmentedColormap.from_list('blue_white_orange', ['#1E40AF','#DBEAFE','white','#FEF3C7','#92400E'])
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, ax=ax,
    annot=True, fmt='.2f', annot_kws={'size':7.5}, square=True,
    linewidths=0.5, linecolor='white', cbar_kws={'shrink':0.8})
ax.set_title('Sparse Feature Correlation Heatmap (k=16 L1-selected features)', pad=12, fontsize=12, fontweight='bold')
ax.tick_params(axis='x', labelsize=8, rotation=45); ax.tick_params(axis='y', labelsize=8, rotation=0)
plt.tight_layout()
plt.savefig(f'{OUT}/fig14_correlation_heatmap.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig14 saved')

# ============================================================
# FIG 15: SHAP per-window beeswarm scatter
# ============================================================
shap_df2 = pd.read_csv('data/processed/full/shap_per_window.csv')
shap_cols = [c for c in shap_df2.columns if c.startswith('SHAP_')][:8]
feat_labels = [c.replace('SHAP_','').replace('frontend__','fe·').replace('order__','ord·').replace('inventory__','inv·').replace('cart__','crt·').replace('shipping__','shp·').replace('response_time_p99','resp').replace('db_connections_active','db_con').replace('http_error_rate','http_e').replace('cpu_usage_percent','cpu').replace('thread_pool_queue','thrd_q') for c in shap_cols]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax2, col, label in zip(axes.flat, shap_cols, feat_labels):
    vals = shap_df2[col].values
    labels = shap_df2['label'].values
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
    ax2.scatter(vals[labels==0], jit[labels==0], c=BLUE, alpha=0.3, s=12, label='Normal')
    ax2.scatter(vals[labels==1], jit[labels==1], c=RED, alpha=0.8, s=40, marker='*', label='Anomaly', zorder=5)
    ax2.axvline(0, color='#374151', linewidth=0.8, linestyle='--')
    ax2.set_title(label, fontsize=8.5, fontweight='bold')
    ax2.set_xlabel('SHAP value', fontsize=8); ax2.set_yticks([])
    ax2.tick_params(labelsize=7.5)
fig.suptitle('SHAP Value Distribution per Feature — Normal vs Anomaly Windows', fontsize=12, fontweight='bold')
from matplotlib.lines import Line2D
legend_handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=BLUE,markersize=8,label='Normal'),
    Line2D([0],[0],marker='*',color='w',markerfacecolor=RED,markersize=10,label='Anomaly')]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5,-0.02))
plt.tight_layout()
plt.savefig(f'{OUT}/fig15_shap_beeswarm.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig15 saved')

# ============================================================
# FIG 16: Mega summary figure (paper-ready)
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

# Panel A: ROC
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot([0,1],[0,1],'--',color='#9CA3AF',linewidth=0.8)
for name,d in roc_data.items():
    lw=2.5 if 'SparseRank' in name else 1.2
    ls='-'
    ax_a.plot(d['fpr'],d['tpr'],color=d['color'],linewidth=lw,linestyle=ls,label=f"AUC={d['auc']:.2f}")
ax_a.set_title('(a) ROC Curves',fontsize=10,fontweight='bold'); ax_a.set_xlabel('FPR',fontsize=9); ax_a.set_ylabel('TPR',fontsize=9)
ax_a.legend(fontsize=6.5,loc='lower right',framealpha=0.8)

# Panel B: PR
ax_b = fig.add_subplot(gs[0, 1])
ax_b.axhline(cont,color='#9CA3AF',linewidth=0.8,linestyle='--')
for name,d in pr_data.items():
    lw=2.5 if 'SparseRank' in name else 1.2
    ax_b.plot(d['rec'],d['prec'],color=d['color'],linewidth=lw,label=f"AP={d['ap']:.2f}")
ax_b.set_title('(b) Precision-Recall',fontsize=10,fontweight='bold'); ax_b.set_xlabel('Recall',fontsize=9); ax_b.set_ylabel('Precision',fontsize=9)
ax_b.legend(fontsize=6.5,loc='upper right',framealpha=0.8)

# Panel C: F1 vs k
ax_c = fig.add_subplot(gs[0, 2])
ax_c.plot(ks, sweep['ANOVA'],'o-',color=GRN,linewidth=1.8,markersize=4,label='ANOVA')
ax_c.plot(ks, sweep['MutualInfo'],'s--',color=TEAL,linewidth=1.5,markersize=4,label='MI')
ax_c.plot(ks, sweep['Random'],'^:',color=GRY,linewidth=1.2,markersize=4,alpha=0.7,label='Random')
ax_c.axhline(results[0]['F1'],color=GRY,linewidth=1,linestyle='-.',alpha=0.6,label='Full IF')
ax_c.scatter([16],[results[1]['F1']],color=BLUE,s=100,zorder=6,marker='*',label='L1 k=16')
ax_c.set_title('(c) F1 vs k',fontsize=10,fontweight='bold'); ax_c.set_xlabel('k features',fontsize=9); ax_c.set_ylabel('F1',fontsize=9)
ax_c.legend(fontsize=7,framealpha=0.8)

# Panel D: Cost vs F1
ax_d = fig.add_subplot(gs[0, 3])
for r in results:
    sz=200 if 'SparseRank' in r['name'] else 80
    ax_d.scatter(r['cost_red'],r['F1'],s=sz,color=r['color'],alpha=0.85,
        edgecolors='white' if 'SparseRank' in r['name'] else r['color'],linewidths=1.5 if 'SparseRank' in r['name'] else 0)
ax_d.set_title('(d) Cost vs F1',fontsize=10,fontweight='bold'); ax_d.set_xlabel('Cost reduction %',fontsize=9); ax_d.set_ylabel('F1',fontsize=9)

# Panel E: SHAP bar
ax_e = fig.add_subplot(gs[1, :2])
top8 = shap_df.head(8)
feat_s = [f.replace('frontend__','fe·').replace('order__','ord·').replace('inventory__','inv·').replace('cart__','crt·').replace('shipping__','shp·').replace('response_time_p99','resp_p99').replace('db_connections_active','db_conn').replace('http_error_rate','http_err').replace('cpu_usage_percent','cpu').replace('thread_pool_queue','thrd_q') for f in top8['feature']]
cols8 = [ORG if 'frontend' in f else BLUE if 'order' in f else GRN if 'inventory' in f else PUR for f in top8['feature']]
ax_e.barh(range(len(top8)), top8['shap_mean'], color=cols8, edgecolor='white', height=0.65)
ax_e.set_yticks(range(len(top8))); ax_e.set_yticklabels(feat_s, fontsize=8.5)
ax_e.set_title('(e) SHAP Feature Importance (top 8)',fontsize=10,fontweight='bold')
ax_e.set_xlabel('Mean |SHAP|',fontsize=9)

# Panel F: Confusion matrix — SparseRank
ax_f = fig.add_subplot(gs[1, 2])
cm_sr = results[1]['cm']
sns.heatmap(cm_sr,annot=True,fmt='d',cmap='Blues',ax=ax_f,cbar=False,annot_kws={'size':14,'weight':'bold'},linewidths=1,linecolor='white')
ax_f.set_xticklabels(['Pred N','Pred A'],fontsize=8); ax_f.set_yticklabels(['True N','True A'],fontsize=8,rotation=0)
ax_f.set_title('(f) CM — SparseRank L1',fontsize=10,fontweight='bold',color=BLUE)

# Panel G: Confusion matrix — Full IF
ax_g = fig.add_subplot(gs[1, 3])
cm_full = results[0]['cm']
sns.heatmap(cm_full,annot=True,fmt='d',cmap='Oranges',ax=ax_g,cbar=False,annot_kws={'size':14,'weight':'bold'},linewidths=1,linecolor='white')
ax_g.set_xticklabels(['Pred N','Pred A'],fontsize=8); ax_g.set_yticklabels(['True N','True A'],fontsize=8,rotation=0)
ax_g.set_title('(g) CM — Full IF (p=228)',fontsize=10,fontweight='bold',color=GRY)

# Panel H: Box plots
ax_h = fig.add_subplot(gs[2, :2])
bp_data_n=[]; bp_data_a=[]; bp_labels=[]
for name,idx,col in dist_cfgs:
    if_d=IsolationForest(n_estimators=100,contamination=min(cont,0.49),random_state=42)
    if_d.fit(X_tr_sc[:,idx]); sc5=if_d.decision_function(X_te_sc[:,idx])
    bp_data_n.append(sc5[y_te==0]); bp_data_a.append(sc5[y_te==1])
    bp_labels.append(name.replace('ANOVA ','k=').replace('SparseRank L1','L1★'))
x_pos = np.arange(len(bp_labels))*2
for i,(n,a) in enumerate(zip(bp_data_n,bp_data_a)):
    bp_n=ax_h.boxplot(n,positions=[x_pos[i]-0.3],widths=0.5,patch_artist=True,notch=False,
        medianprops={'color':'white','lw':2},boxprops={'facecolor':'#93C5FD','alpha':0.6},flierprops={'marker':'.','markersize':3,'alpha':0.4})
    bp_a=ax_h.boxplot(a,positions=[x_pos[i]+0.3],widths=0.5,patch_artist=True,notch=False,
        medianprops={'color':'white','lw':2},boxprops={'facecolor':'#FCA5A5','alpha':0.7},flierprops={'marker':'.','markersize':3,'alpha':0.5})
ax_h.set_xticks(x_pos); ax_h.set_xticklabels(bp_labels,fontsize=8.5,rotation=15,ha='right')
ax_h.axhline(0,color='#374151',linewidth=0.8,linestyle='--',alpha=0.5)
ax_h.set_ylabel('IF Score',fontsize=9); ax_h.set_title('(h) Box Plots — Score Distributions by Feature Class',fontsize=10,fontweight='bold')

# Panel I: CV
ax_i = fig.add_subplot(gs[2, 2:])
for mn,d in cv_res.items():
    col=cv_colors[mn]; mk=markers[mn]
    ax_i.plot(range(1,6),d['f1'],f'{mk}-',color=col,linewidth=2,markersize=6,label=f"{mn} μ={np.mean(d['f1']):.3f}")
    ax_i.fill_between(range(1,6),[v-np.std(d['f1'])*0.5 for v in d['f1']],[v+np.std(d['f1'])*0.5 for v in d['f1']],color=col,alpha=0.07)
ax_i.set_xlabel('Fold',fontsize=9); ax_i.set_ylabel('F1',fontsize=9)
ax_i.set_title('(i) 5-Fold Cross-Validation',fontsize=10,fontweight='bold')
ax_i.legend(fontsize=8.5); ax_i.set_xticks(range(1,6)); ax_i.set_xticklabels([f'F{i}' for i in range(1,6)])

fig.suptitle('SparseRank: Complete Results Summary — EBISION 2026', fontsize=14, fontweight='bold', y=1.01)
plt.savefig(f'{OUT}/fig16_complete_summary.png', bbox_inches='tight', dpi=150)
plt.close(); print('  fig16 saved')

print(f'\nAll figures saved to {OUT}/')
for fn in sorted(os.listdir(OUT)):
    if fn.endswith('.png'):
        sz = os.path.getsize(f'{OUT}/{fn}')//1024
        print(f'  {fn}  ({sz}KB)')
