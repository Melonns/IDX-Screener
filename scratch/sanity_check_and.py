"""
Sanity check untuk Tahap 2b (AND intersection):
Berapa sinyal yang keluar di berbagai rank threshold?

Ini bukan backtest penuh — hanya hitung jumlah sinyal yang lolos filter
untuk kalibrasi search space dan constraint minimum sample.
"""
import sys, os
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from data.database import DatabaseManager
import config as app_config

db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))
MIN_TURNOVER = 1_000_000_000
TRAINING_END = '2026-02-09'

print("Loading contextual_indicators (training set)...")
with db._connect() as conn:
    df = pd.read_sql_query(f"""
        SELECT date, ticker,
               rel_strength_5d_rank, vol_accum_5d_rank, turnover_5d
        FROM contextual_indicators
        WHERE date <= '{TRAINING_END}'
        AND rel_strength_5d_rank IS NOT NULL
        AND vol_accum_5d_rank IS NOT NULL
        AND turnover_5d IS NOT NULL
    """, conn)

# Filter turnover dulu (syarat mutlak)
df_liquid = df[df['turnover_5d'] >= MIN_TURNOVER].copy()

total_liquid_obs = len(df_liquid)
n_days = df_liquid['date'].nunique()
n_tickers = df_liquid['ticker'].nunique()

print(f"\nData setelah turnover filter:")
print(f"  Observasi liquid       : {total_liquid_obs:,}")
print(f"  Hari trading           : {n_days}")
print(f"  Tickers                : {n_tickers}")
print(f"  Rata-rata liquid/hari  : {total_liquid_obs/n_days:.1f} saham/hari")

# Benchmark: single-feature (OR) di threshold 15%
bench_rs  = df_liquid[df_liquid['rel_strength_5d_rank'] <= 15.0]
bench_vol = df_liquid[df_liquid['vol_accum_5d_rank'] <= 15.0]
print(f"\n  --- Benchmark Single-Feature @ rank=15% ---")
print(f"  rel_strength alone : {len(bench_rs):,} sinyal total | {len(bench_rs)/n_days:.2f}/hari")
print(f"  vol_accum alone    : {len(bench_vol):,} sinyal total | {len(bench_vol)/n_days:.2f}/hari")

# AND intersection di berbagai threshold
print(f"\n{'='*65}")
print(f"  AND INTERSECTION — Signal Count per Threshold")
print(f"  (Butuh minimal 30+ sinyal per fold = 150+ total buat reliable)")
print(f"{'='*65}")
print(f"{'rank_thresh':>12} | {'total_signals':>13} | {'per_day':>9} | {'per_fold_est':>13} | {'reliable?':>10}")
print(f"{'-'*65}")

thresholds = [10, 15, 20, 25, 30, 35, 40]
for thresh in thresholds:
    mask = (df_liquid['rel_strength_5d_rank'] <= thresh) & (df_liquid['vol_accum_5d_rank'] <= thresh)
    n = mask.sum()
    per_day = n / n_days
    per_fold_est = n / 5  # 5 fold
    reliable = "✓ OK" if per_fold_est >= 30 else ("⚠ TIPIS" if per_fold_est >= 10 else "✗ TERLALU SEDIKIT")
    print(f"{thresh:>10}%  | {n:>13,} | {per_day:>9.2f} | {per_fold_est:>13.0f} | {reliable:>10}")

# Lihat juga berapa expected unique sinyal (per hari, bukan per-ticker-day)
print(f"\n  --- Detail Distribusi Sinyal per Hari @ beberapa threshold ---")
for thresh in [20, 25, 30]:
    mask = (df_liquid['rel_strength_5d_rank'] <= thresh) & (df_liquid['vol_accum_5d_rank'] <= thresh)
    daily = df_liquid[mask].groupby('date').size()
    print(f"\n  rank={thresh}%:")
    print(f"    Hari dengan 0 sinyal    : {(daily.reindex(df_liquid['date'].unique(), fill_value=0) == 0).sum()}")
    print(f"    Hari dengan 1+ sinyal   : {(daily > 0).sum()}")
    print(f"    Hari dengan 2+ sinyal   : {(daily >= 2).sum()}")
    print(f"    Hari dengan 5+ sinyal   : {(daily >= 5).sum()}")
    print(f"    Rata-rata sinyal/hari   : {daily.mean():.2f}")
    print(f"    Median sinyal/hari      : {daily.median():.1f}")
