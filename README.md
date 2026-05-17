# SparseRank

**Explainable Cost-Aware Feature Selection for Anomaly Detection in E-Commerce Microservices**

EBISION 2026 — IFIP WG 8.4 International Symposium on E-Business Information Systems Evolution


## 1. What this project does

SparseRank is a three-stage pipeline that answers two operational questions for e-commerce SREs:

## 2. Prerequisites

Make sure the following are installed before starting.

| Tool | Minimum version | How to check |
|------|----------------|--------------|
| Python | 3.10 | `python --version` |
| pip | 23.0 | `pip --version` |
| Git | 2.x | `git --version` |
| VS Code | 1.85 | Help → About |

**Recommended VS Code extensions** — install from the Extensions panel (`Ctrl+Shift+X`):

- **Python** (Microsoft) — `ms-python.python`
- **Pylance** (Microsoft) — `ms-python.vscode-pylance`
- **Jupyter** (Microsoft) — `ms-toolsai.jupyter`
- **Rainbow CSV** — `mechatroner.rainbow-csv` (for browsing output CSVs)

---

## 3. Clone and open in VS Code

Open a terminal and run:

```bash
git clone https://github.com/your-username/sparserank.git
cd sparserank
code .
```

> If you downloaded the project as a ZIP instead, extract it then open the folder in VS Code with **File → Open Folder**.

---

## 4. Create and activate a virtual environment

Open the integrated terminal in VS Code with `` Ctrl+` `` then run the commands for your OS.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you get a script execution policy error, run this first, then retry activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Once activated, your terminal prompt will show `(.venv)` at the start.
**All subsequent commands must be run with the virtual environment active.**

---

## 5. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `pandas >= 2.0` | Data loading and CSV output |
| `numpy >= 1.24` | Array operations |
| `scikit-learn >= 1.4` | LogisticRegression (L1), IsolationForest, PCA, t-SNE, metrics |
| `shap >= 0.43` | TreeSHAP explainability |
| `matplotlib >= 3.8` | Figure generation |
| `seaborn >= 0.13` | Heatmaps and styled plots |
| `scipy >= 1.11` | Spearman rank correlation |
| `pyarrow >= 14.0` | Parquet caching for fast re-runs |

Installation takes ~2–3 minutes. Verify with:

```bash
python -c "import sklearn, shap, pandas, numpy, matplotlib; print('All packages OK')"
```

---

## 6. Configure VS Code Python interpreter

Press `Ctrl+Shift+P` → type **Python: Select Interpreter** → choose the entry that shows `.venv` in its path:

```
Python 3.11.x ('.venv': venv)  ./venv/bin/python
```

This ensures VS Code uses the virtual environment for IntelliSense and when you run scripts from the editor.

---

## 7. Project structure

```
sparserank/
│
├── src/                        ← source modules
│   ├── __init__.py
│   ├── data_loader.py          ← generates / loads the dataset
│   ├── feature_select.py       ← L1 sweep, picks k features
│   ├── anomaly_detector.py     ← IsolationForest + KPI validation
│   ├── shap_explainer.py       ← TreeSHAP + rank correlation
│   ├── evaluate.py             ← baselines, tables, 4-panel figure
│   └── full_analysis.py        ← extended: PCA, t-SNE, CV, all CSVs
│
├── data/
│   ├── rs_anomic/              ← place real RS-Anomic CSVs here (optional)
│   └── processed/              ← auto-generated: .parquet, .npy, .pkl, .csv
│       └── full/               ← extended analysis outputs
│
├── experiments/
│   └── figures/                ← all generated PNG figures
│
├── paper/
│   └── sparserank_draft.md     ← conference paper draft
│
├── run_pipeline.py             ← MAIN ENTRY POINT
├── generate_figures.py         ← generates all 16 publication figures
├── requirements.txt
└── README.md
```

---

## 8. Run the full pipeline

This single command runs all four stages in order and is the recommended starting point:

```bash
python run_pipeline.py
```


### Expected terminal output

```
============================================================
  SparseRank — Full Pipeline
============================================================

[STEP 2] L1-regularized feature selection
  ✓ Best C=0.01: k=24, cost reduction=94.0%

## 9. Run individual stages

All commands must be run from the project root (`sparserank/` directory).

```bash
# Stage 1 — data loader only
python -c "import sys; sys.path.insert(0,'.'); from src.data_loader import load_dataset; df=load_dataset(); print(df.shape)"

# Stage 2 — feature selection
python -c "import sys,os; sys.path.insert(0,'.'); os.chdir('.'); import src.feature_select as m; m.main()"

# Stage 3 — anomaly detection
python -c "import sys,os; sys.path.insert(0,'.'); os.chdir('.'); import src.anomaly_detector as m; m.main()"

# Stage 4 — SHAP explainability
python -c "import sys,os; sys.path.insert(0,'.'); os.chdir('.'); import src.shap_explainer as m; m.main()"

# Stage 5 — evaluation and figures
python -c "import sys,os; sys.path.insert(0,'.'); os.chdir('.'); import src.evaluate as m; m.main()"
```

```

### Key CSV columns

**`feature_select_results.csv`**

| Column | Description |
|--------|-------------|
| `C` | Regularisation strength |
| `k` | Number of selected features |
| `F1` | F1 on the test set |
| `cost_red_pct` | Monitoring cost reduction (%) |

**`eval_metrics.csv`** — single row for the final SparseRank model with F1, Precision, Recall, ROC_AUC, AP, MCC, and KPI Spearman rho.

**`baseline_comparison.csv`** — one row per method with all metrics including TP, FP, TN, FN. Use this directly in the paper results table.

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

You are not in the project root directory. Run:

```bash
cd sparserank
python run_pipeline.py
```

### `FileNotFoundError: data/processed/selected_idx.npy`

Stage 2 has not run yet. Run the full pipeline first:

```bash
python run_pipeline.py
```

### `TSNE.__init__() got an unexpected keyword argument 'n_iter'`

Upgrade scikit-learn:

```bash
pip install --upgrade scikit-learn
```

### VS Code shows red underlines on all imports

The wrong Python interpreter is selected. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `.venv` entry.

### First run is slow

The first run generates the synthetic dataset and trains all models from scratch. Subsequent runs use the cached parquet file and complete in under 10 seconds. To force a full regeneration:

```bash
rm data/processed/rs_anomic_wide.parquet    # macOS / Linux
del data\processed\rs_anomic_wide.parquet   # Windows
```

### `pip install` fails with a Visual C++ error (Windows only)

Install Microsoft C++ Build Tools from:
`https://visualstudio.microsoft.com/visual-cpp-build-tools/`

Then retry installation.

### Figures are blank or missing data

Run the full pipeline before `generate_figures.py`. All `.npy`, `.pkl`, and `.csv` files in `data/processed/` must exist first.

---

## Using real RS-Anomic data (optional)

Place the RS-Anomic CSV files inside `data/rs_anomic/` in this structure:

```
data/rs_anomic/
├── normal_data/
│   └── <service>_<metric>.csv
└── anomaly_data/
    └── <service>_<metric>.csv
```

Then delete the cache and re-run. `data_loader.py` detects the CSV files automatically and loads them instead of generating synthetic data.

```bash
rm data/processed/rs_anomic_wide.parquet
python run_pipeline.py


*SparseRank · EBISION 2026 · Python 3.10+ · CPU only · ~40 s end-to-end*
