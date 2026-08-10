"""
screening_stage3.py — Quick-and-Dirty Screening untuk Tahap 3A (Confirmation) & 3B (Regime)

Tujuan: Fail Fast — Cek apakah Confirmation Layer (3A) atau Regime Filter (3B)
bisa mendongkrak Gross EV vol_accum_5d (+0.230%) melewati threshold breakeven 0.40%.

Tanpa Optuna 50 trials — screening cepat di Training Set (max_date='2026-02-09').
"""
import sys, os
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from data.database import DatabaseManager
from scoring.contextual_engine import ContextualEngine
from backtest.contextual_backtest import ContextualWalkForwardBacktester
from backtest.engine import IDX_ROUNDTRIP_COST
import config as app_config

db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))
tickers = db.get_tickers()
TRAINING_END = '2026-02-09'

# Benchmark Base: vol_accum_5d @ rank=14.9% (Tahap 2a best)
engine_base = ContextualEngine(rank_threshold=14.9, feature_mode='vol_accum')
backtester = ContextualWalkForwardBacktester(
    db=db, engine=engine_base, n_folds=5, max_date=TRAINING_END, target_threshold_pct=2.0
)
res_base = backtester.run(tickers)
all_base = [s for f in res_base.folds for s in f.bullish_signals if s.return_n3 is not None]

print("\n" + "="*70)
print("  BENCHMARK TAHAP 2a: vol_accum_5d Standalone (rank=14.9%)")
print("="*70)
rets_base = [s.return_n3 for s in all_base]
gross_base = np.mean(rets_base) if rets_base else 0
print(f"  Total sinyal   : {len(all_base)}")
print(f"  Gross EV       : {gross_base*100:+.4f}%")
print(f"  Net EV (0.40%) : {(gross_base - IDX_ROUNDTRIP_COST)*100:+.4f}%")

# Load full context dataset for detailed candle & regime matching
print("\n" + "="*70)
print("  TAHAP 3A: CONFIRMATION LAYER SCREENING (Price / Volume Action)")
print("="*70)

# Build a fast lookup dataframe for all signals to test 3A & 3B filters
records = []
for f in res_base.folds:
    for s in f.bullish_signals:
        if s.return_n3 is None: continue
        records.append({
            'ticker': s.ticker,
            'date': s.date,
            'entry_price': s.entry_price,
            'return_n3': s.return_n3,
            'fold': f.fold_idx
        })

sig_df = pd.DataFrame(records)

# Fetch price and context data for confirmation & regime checks
with db._connect() as conn:
    prices = pd.read_sql_query(f"""
        SELECT p.date, p.ticker, p.open, p.close, p.high, p.low, p.volume,
               c.vol_accum_5d, c.vol_accum_5d_rank, c.turnover_5d,
               m.slope_20d AS ihsg_slope_20d, m.ret_5d AS ihsg_ret_5d
        FROM daily_prices p
        LEFT JOIN contextual_indicators c USING (ticker, date)
        LEFT JOIN market_index m ON p.date = m.date
        WHERE p.date <= '{TRAINING_END}' AND p.is_valid = 1
    """, conn)

merged = pd.merge(sig_df, prices, on=['ticker', 'date'], how='left')

# Calculate candle indicators
merged['is_green_candle'] = merged['close'] > merged['open']
merged['is_up_day']       = merged['close'] > merged['open']  # 1d return > 0 approx

# ── 3A Screening Filters ──
filters_3a = {
    'Base (Tanpa filter)': merged,
    '3A.1: Green Candle (Close > Open)': merged[merged['is_green_candle'] == True],
    '3A.2: Red Candle (Close < Open)': merged[merged['close'] < merged['open']],
    '3A.3: Body > 0.5% (Konfirmasi bullish)': merged[(merged['close'] - merged['open']) / merged['open'] > 0.005],
}

print(f"{'Filter 3A':<40} | {'N Sinyal':>8} | {'Gross EV':>9} | {'Net EV (0.4%)':>12} | {'vs Base':>8}")
print("-" * 80)

for name, f_df in filters_3a.items():
    n = len(f_df)
    if n == 0: continue
    g_ev = f_df['return_n3'].mean()
    n_ev = g_ev - IDX_ROUNDTRIP_COST
    diff = g_ev - gross_base
    print(f"{name:<40} | {n:>8,d} | {g_ev*100:>+8.4f}% | {n_ev*100:>+11.4f}% | {diff*100:>+7.4f}%")

# ── 3B Screening Filters (Market Regime) ──
print("\n" + "="*70)
print("  TAHAP 3B: MARKET REGIME SCREENING (IHSG Slope / Trend)")
print("="*70)

filters_3b = {
    'Base (Tanpa regime filter)': merged,
    '3B.1: IHSG Uptrend (slope_20d > 0)': merged[merged['ihsg_slope_20d'] > 0],
    '3B.2: IHSG Downtrend (slope_20d < 0)': merged[merged['ihsg_slope_20d'] < 0],
    '3B.3: IHSG Strong Uptrend (slope > 0.001)': merged[merged['ihsg_slope_20d'] > 0.001],
    '3B.4: IHSG Strong Downtrend (slope < -0.001)': merged[merged['ihsg_slope_20d'] < -0.001],
}

print(f"{'Filter 3B (Regime)':<40} | {'N Sinyal':>8} | {'Gross EV':>9} | {'Net EV (0.4%)':>12} | {'vs Base':>8}")
print("-" * 80)

for name, f_df in filters_3b.items():
    n = len(f_df)
    if n == 0: continue
    g_ev = f_df['return_n3'].mean()
    n_ev = g_ev - IDX_ROUNDTRIP_COST
    diff = g_ev - gross_base
    print(f"{name:<40} | {n:>8,d} | {g_ev*100:>+8.4f}% | {n_ev*100:>+11.4f}% | {diff*100:>+7.4f}%")

# ── Combined 3A + 3B ──
print("\n" + "="*70)
print("  TAHAP 3A + 3B COMBINED SCREENING")
print("="*70)

combos = {
    '3A.1 + 3B.1 (Green Candle + IHSG Uptrend)': merged[(merged['is_green_candle'] == True) & (merged['ihsg_slope_20d'] > 0)],
    '3A.1 + 3B.2 (Green Candle + IHSG Downtrend)': merged[(merged['is_green_candle'] == True) & (merged['ihsg_slope_20d'] < 0)],
    '3A.3 + 3B.1 (Strong Body + IHSG Uptrend)': merged[((merged['close'] - merged['open'])/merged['open'] > 0.005) & (merged['ihsg_slope_20d'] > 0)],
}

print(f"{'Filter Kombinasi':<45} | {'N Sinyal':>8} | {'Gross EV':>9} | {'Net EV (0.4%)':>12} | {'vs Base':>8}")
print("-" * 85)

for name, f_df in combos.items():
    n = len(f_df)
    if n == 0: continue
    g_ev = f_df['return_n3'].mean()
    n_ev = g_ev - IDX_ROUNDTRIP_COST
    diff = g_ev - gross_base
    print(f"{name:<45} | {n:>8,d} | {g_ev*100:>+8.4f}% | {n_ev*100:>+11.4f}% | {diff*100:>+7.4f}%")
