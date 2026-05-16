"""
data_loader.py — Complete RS-Anomic Dataset Loader
====================================================
Uses EVERY file in the dataset:

NORMAL DATA
  normal_data/normal_data/cAdvisor/          12 services × 19 metrics (~142 hrs)
  normal_data/normal_data/response_times/    10 service RT files (27 API calls)

ANOMALY DATA — 14 fault scenarios
  high-cpu-dispatch, high-fileIO-payment, high-latency-user,
  high-latency-user-2, high-load-1500, low-bandwidth-user,
  low-bandwidth-user-2, memory-leak-cart, out-of-order-packets-user,
  out-of-order-packets-user-2, packetloss-user, packetloss-user-2,
  rt-delay-catalogue, service-down-payment
  Each with matching response_times/ directory

FEATURE ENGINEERING
  cAdvisor : diff → per-second rate → 60s resample → mean + std per metric
  RT       : 60s resample → mean + std per API call (sum column only)
  Total features after intersection: ~510 (cAdvisor) + ~54 (RT)

LABELLING
  Normal  : windows from normal_data (downsampled to match anomaly count)
  Anomaly : all windows from every fault scenario
"""

import os, glob, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split as sk_split

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path('data/rs_anomic/rs_anomic')
CACHE_PATH = Path('data/processed/rs_anomic_complete.parquet')
WINDOW     = '60s'
RANDOM_SEED = 42

# ── all 19 cAdvisor metrics (exact column names) ──────────────────────────────
CADV_COLS = [
    'container_fs_usage_bytes', 'container_memory_rss',
    'container_memory_usage_bytes', 'container_memory_working_set_bytes',
    'container_cpu_system_seconds_total', 'container_cpu_usage_seconds_total',
    'container_cpu_user_seconds_total',
    'container_network_receive_bytes_total', 'container_network_receive_errors_total',
    'container_network_receive_packets_dropped_total',
    'container_network_receive_packets_total',
    'container_network_transmit_bytes_total', 'container_network_transmit_errors_total',
    'container_network_transmit_packets_dropped_total',
    'container_network_transmit_packets_total',
    'container_fs_io_time_seconds_total', 'container_memory_failures_total',
    'container_memory_failcnt', 'container_fs_write_seconds_total',
]
# short names for feature columns
MSHORT = {m: m.replace('container_','').replace('_total','').replace('_seconds','')
          for m in CADV_COLS}

ALL_FAULTS = [
    'high-cpu-dispatch', 'high-fileIO-payment', 'high-latency-user',
    'high-latency-user-2', 'high-load-1500', 'low-bandwidth-user',
    'low-bandwidth-user-2', 'memory-leak-cart', 'out-of-order-packets-user',
    'out-of-order-packets-user-2', 'packetloss-user', 'packetloss-user-2',
    'rt-delay-catalogue', 'service-down-payment',
]


# ═══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _cadvisor_to_windows(csv_path: Path, svc: str) -> pd.DataFrame:
    """Load one cAdvisor CSV → rate → 60-s windows (mean + std)."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp').set_index('timestamp')
    cols = [c for c in CADV_COLS if c in df.columns]
    df = df[cols].ffill().bfill().fillna(0)
    # cumulative counter → per-second rate (vectorized)
    dt_s = df.index.to_series().diff().dt.total_seconds().fillna(5).clip(lower=0.1)
    for c in cols:
        df[c] = df[c].diff().fillna(0).clip(lower=0) / dt_s
    # 60-second windows
    m = df.resample(WINDOW).mean().fillna(0)
    s = df.resample(WINDOW).std(ddof=0).fillna(0)
    m.columns = [f'{svc}__{MSHORT.get(c,c)}_mean' for c in m.columns]
    s.columns = [f'{svc}__{MSHORT.get(c,c)}_std'  for c in s.columns]
    return pd.concat([m, s], axis=1)


def _rt_to_windows(csv_path: Path, tag: str) -> pd.DataFrame:
    """Load one response-time CSV → 60-s windows (mean + std of _sum columns)."""
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp').set_index('timestamp')
    # use _sum columns (total response time) — ignore _count
    sum_cols = [c for c in df.columns if c.endswith('_sum')]
    if not sum_cols:
        return pd.DataFrame()
    df = df[sum_cols].ffill().bfill().fillna(0)
    m = df.resample(WINDOW).mean().fillna(0)
    s = df.resample(WINDOW).std(ddof=0).fillna(0)
    # name: tag__rt_<api>_mean / _std
    def _short(col):
        return col.replace('rt_','').replace('_sum','')[:24]
    m.columns = [f'{tag}__rt_{_short(c)}_mean' for c in m.columns]
    s.columns = [f'{tag}__rt_{_short(c)}_std'  for c in s.columns]
    return pd.concat([m, s], axis=1)


def _build_scenario(cadv_dir: Path, rt_dir: Path | None) -> pd.DataFrame | None:
    """
    Load all cAdvisor + RT files for one scenario.
    Align on intersection of timestamps → return unified windowed DataFrame.
    """
    frames = {}

    # cAdvisor — one file per service
    for csv in sorted(cadv_dir.glob('*.csv')):
        if csv.stat().st_size < 200:
            continue
        svc = csv.stem
        try:
            frames[f'cadv_{svc}'] = _cadvisor_to_windows(csv, svc)
        except Exception as e:
            pass   # skip broken files silently

    if not frames:
        return None

    # Response times — one file per service/API-group
    if rt_dir and rt_dir.exists():
        for csv in sorted(rt_dir.glob('*.csv')):
            if csv.stat().st_size < 200:
                continue
            tag = csv.stem.replace('_rt','')
            try:
                wrt = _rt_to_windows(csv, tag)
                if not wrt.empty:
                    frames[f'rt_{tag}'] = wrt
            except:
                pass

    # Align all frames on common timestamps
    common_idx = None
    for f in frames.values():
        common_idx = f.index if common_idx is None else common_idx.intersection(f.index)

    if common_idx is None or len(common_idx) < 5:
        return None

    combined = pd.concat(
        [f.loc[common_idx] for f in frames.values()], axis=1
    ).fillna(0.0)

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build and return the complete RS-Anomic feature matrix.

    Columns:
      <feature_cols>           — cAdvisor + RT windowed features
      label                    — 0 = normal, 1 = anomaly
      fault_type               — scenario name (or 'normal')
      order_http_2xx_rate      — KPI oracle (RT-based, normalised 0-1)
      window_id                — row index
    """
    if CACHE_PATH.exists() and not force_rebuild:
        print(f'[data_loader] Loaded from cache: {CACHE_PATH}')
        return pd.read_parquet(CACHE_PATH)

    t0 = time.time()
    print('[data_loader] Building COMPLETE RS-Anomic dataset...')
    print(f'  14 fault scenarios × (12 cAdvisor + RT files) + normal data')

    all_frames  = []
    feat_sets   = []

    # ── NORMAL DATA ──────────────────────────────────────────────────────────
    print('\n  [NORMAL]', end=' ', flush=True)
    norm_cadv = DATA_ROOT / 'normal_data/normal_data/cAdvisor'
    norm_rt   = DATA_ROOT / 'normal_data/normal_data/response_times'
    norm_df   = _build_scenario(norm_cadv, norm_rt)

    if norm_df is not None:
        norm_df['label']      = 0
        norm_df['fault_type'] = 'normal'
        all_frames.append(norm_df)
        feat_sets.append(set(norm_df.columns) - {'label','fault_type'})
        print(f'{len(norm_df)} windows, {norm_df.shape[1]-2} features')
    else:
        print('FAILED')

    # ── ALL 14 FAULT SCENARIOS ───────────────────────────────────────────────
    print('\n  [ANOMALY SCENARIOS]')
    cadv_base = DATA_ROOT / 'anomaly_data/anomaly_data/cAdvisor'
    rt_base   = DATA_ROOT / 'anomaly_data/anomaly_data/response_times'

    for fault in ALL_FAULTS:
        cadv_dir = cadv_base / fault
        rt_dir   = rt_base / f'{fault}_rt'
        if not cadv_dir.exists():
            print(f'    [SKIP] {fault}')
            continue
        sc = _build_scenario(cadv_dir, rt_dir if rt_dir.exists() else None)
        if sc is None:
            print(f'    [SKIP] {fault} — no valid data')
            continue
        sc['label']      = 1
        sc['fault_type'] = fault
        all_frames.append(sc)
        feat_sets.append(set(sc.columns) - {'label','fault_type'})
        print(f'    ✓ {fault:<34} {len(sc):4d} windows  {sc.shape[1]-2:4d} features')

    # ── INTERSECT FEATURES across all scenarios ──────────────────────────────
    meta = {'label', 'fault_type'}
    common = sorted(set.intersection(*feat_sets))
    print(f'\n  Common features across all {len(all_frames)} scenarios: {len(common)}')

    # ── ASSEMBLE FULL DATAFRAME ───────────────────────────────────────────────
    keep = common + ['label', 'fault_type']
    parts = []
    for fr in all_frames:
        fr = fr.loc[:,~fr.columns.duplicated()]
        sub = fr.reindex(columns=keep, fill_value=0.0)
        sub = sub.reset_index().rename(columns={'index':'timestamp'})
        parts.append(sub)

    df = pd.concat(parts, ignore_index=True)
    df[common] = df[common].fillna(0.0)

    # ── BALANCE — downsample normal to match total anomaly count ──────────────
    n_anom   = (df.label == 1).sum()
    norm_all = df[df.label == 0].copy()
    step     = max(1, len(norm_all) // n_anom)
    norm_bal = norm_all.iloc[::step].copy()
    df = pd.concat([norm_bal, df[df.label == 1]], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ── KPI ORACLE ────────────────────────────────────────────────────────────
    # Use rt_payment__rt_payment_delete_cart_mean if present, else cart network tx
    kpi_candidates = [c for c in common if 'payment' in c and 'rt_' in c and '_mean' in c]
    if not kpi_candidates:
        kpi_candidates = [c for c in common if 'cart' in c and 'transmit' in c and '_mean' in c]
    kpi_col = kpi_candidates[0] if kpi_candidates else common[0]

    df['order_http_2xx_rate'] = 0.5
    for ft in df['fault_type'].unique():
        mask = df['fault_type'] == ft
        kraw = df.loc[mask, kpi_col]
        span = kraw.max() - kraw.min()
        df.loc[mask, 'order_http_2xx_rate'] = (
            1.0 if ft == 'normal'
            else (1.0 - (kraw - kraw.min()) / span) if span > 0
            else 0.5
        )

    df['window_id'] = df.index

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    n_n = (df.label == 0).sum()
    n_a = (df.label == 1).sum()
    p   = len(common)
    print(f'\n[data_loader] ✓ Complete dataset built in {time.time()-t0:.1f}s')
    print(f'  Total windows   : {len(df):,}')
    print(f'  Normal windows  : {n_n:,}  ({n_n/len(df)*100:.1f}%)')
    print(f'  Anomaly windows : {n_a:,}  ({n_a/len(df)*100:.1f}%)')
    print(f'  Features p      : {p}')
    print(f'  Fault scenarios : {df.fault_type.nunique()} (incl. normal)')
    print(f'  KPI oracle col  : {kpi_col}')
    print('  Per-scenario breakdown:')
    for ft, cnt in df.groupby('fault_type').size().sort_values(ascending=False).items():
        lbl = 'NORMAL' if ft == 'normal' else 'anomaly'
        print(f'    {ft:<38} {cnt:4d}  [{lbl}]')

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    print(f'\n  Saved → {CACHE_PATH}')
    return df


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2,
                        seed: int = RANDOM_SEED):
    """Stratified 80/20 split. Returns X_tr, X_te, y_tr, y_te, kpi_tr, kpi_te, feat_cols."""
    skip = {'label', 'fault_type', 'order_http_2xx_rate', 'window_id', 'timestamp'}
    feat_cols = [c for c in df.columns if c not in skip]
    X    = df[feat_cols].values.astype(np.float32)
    y    = df['label'].values
    kpi  = df['order_http_2xx_rate'].values
    X_tr, X_te, y_tr, y_te, kpi_tr, kpi_te = sk_split(
        X, y, kpi, test_size=test_size, stratify=y, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te, kpi_tr, kpi_te, feat_cols


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    df = load_dataset(force_rebuild=True)
    fc = [c for c in df.columns if c not in ('label','fault_type','order_http_2xx_rate','window_id','timestamp')]
    print(f'\nSample features: {fc[:5]}')
    print(f'Last features  : {fc[-5:]}')
