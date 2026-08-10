"""
benchmark_swing_n20.py — Fast Vectorized Swing Benchmark Test (N+20 Hari)

Bandingkan secara langsung:
1. Strategy: vol_accum_5d (rank <= 14.9%, N+20 days)
2. Baseline 1 (Random Liquid Selection): Rata-rata return N+20 dari SELURUH saham liquid pada hari sinyal.
3. Baseline 2 (IHSG Index): Return N+20 dari IHSG (^JKSE) pada hari sinyal.
"""
import sys, os
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from data.database import DatabaseManager
from backtest.engine import IDX_ROUNDTRIP_COST
import config as app_config

TRAINING_END = '2026-02-09'
RANK_THRESH  = 14.9
MIN_TURNOVER = 1_000_000_000
HORIZON      = 20

db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))

print("Loading full price and index data for fast matrix calculation...")
with db._connect() as conn:
    prices = pd.read_sql_query(f"""
        SELECT p.date, p.ticker, p.close,
               c.vol_accum_5d_rank, c.turnover_5d
        FROM daily_prices p
        LEFT JOIN contextual_indicators c USING (ticker, date)
        WHERE p.date <= '{TRAINING_END}' AND p.is_valid = 1
        ORDER BY p.ticker, p.date ASC
    """, conn)
    
    ihsg = pd.read_sql_query(f"""
        SELECT date, close AS ihsg_close
        FROM market_index
        WHERE symbol = '^JKSE' AND date <= '{TRAINING_END}'
        ORDER BY date ASC
    """, conn)

# Pre-compute N+20 return for EVERY ticker-date pair
prices['date'] = pd.to_datetime(prices['date'])
ihsg['date']   = pd.to_datetime(ihsg['date'])

# Shift 20 trading days forward per ticker
prices['close_n20'] = prices.groupby('ticker')['close'].shift(-HORIZON)
prices['ret_n20']   = (prices['close_n20'] - prices['close']) / prices['close']

# IHSG N+20 return
ihsg['ihsg_n20'] = ihsg['ihsg_close'].shift(-HORIZON)
ihsg['ihsg_ret_n20'] = (ihsg['ihsg_n20'] - ihsg['ihsg_close']) / ihsg['ihsg_close']

# Filter valid training rows with turnover >= 1B
liquid_prices = prices[(prices['turnover_5d'] >= MIN_TURNOVER) & (prices['ret_n20'].notna())].copy()

# Strategy signals (vol_accum <= 14.9)
strat_df = liquid_prices[liquid_prices['vol_accum_5d_rank'] <= RANK_THRESH].copy()

# Merge with daily liquid average (Random Selection Baseline)
daily_random_avg = liquid_prices.groupby('date')['ret_n20'].mean().reset_index()
daily_random_avg.rename(columns={'ret_n20': 'random_ret_n20'}, inplace=True)

# Merge strategy signals with Random Average & IHSG return
merged = pd.merge(strat_df, daily_random_avg, on='date', how='left')
merged = pd.merge(merged, ihsg[['date', 'ihsg_ret_n20']], on='date', how='left')

valid_eval = merged.dropna(subset=['ret_n20', 'random_ret_n20', 'ihsg_ret_n20']).copy()

# Summary Calculations
n = len(valid_eval)
strat_gross = valid_eval['ret_n20'].mean()
strat_net   = strat_gross - IDX_ROUNDTRIP_COST

rand_gross  = valid_eval['random_ret_n20'].mean()
rand_net    = rand_gross - IDX_ROUNDTRIP_COST

ihsg_gross  = valid_eval['ihsg_ret_n20'].mean()
ihsg_net    = ihsg_gross - 0.001  # Assuming 0.1% ETF fee

alpha_vs_random = strat_gross - rand_gross
alpha_vs_ihsg   = strat_gross - ihsg_gross

print("\n" + "="*75)
print("  HASIL BENCHMARK COMPARISON (SWING HORIZON N=20 HARI)")
print("="*75)

print(f"{'Metode / Baseline':<35} | {'N Trade':>8} | {'Gross EV':>9} | {'Net EV':>10} | {'Win Rate (>0)':>12}")
print("-" * 80)
print(f"{'1. Strategi (vol_accum_5d N+20)':<35} | {n:>8,d} | {strat_gross*100:>+8.4f}% | {strat_net*100:>+9.4f}% | {(valid_eval['ret_n20']>0).mean()*100:>11.1f}%")
print(f"{'2. Baseline 1 (Random Liquid N+20)':<35} | {n:>8,d} | {rand_gross*100:>+8.4f}% | {rand_net*100:>+9.4f}% | {(valid_eval['random_ret_n20']>0).mean()*100:>11.1f}%")
print(f"{'3. Baseline 2 (IHSG Index N+20)':<35} | {n:>8,d} | {ihsg_gross*100:>+8.4f}% | {ihsg_net*100:>+9.4f}% | {(valid_eval['ihsg_ret_n20']>0).mean()*100:>11.1f}%")

print("\n" + "="*75)
print("  EVALUASI ALPHA (KEUNGGULAN STRATEGI TERHADAP BENCHMARK)")
print("="*75)
print(f"  Alpha Gross vs Random Selection  : {alpha_vs_random*100:+.4f}% ({'✅ STRATEGI UNGGUL (MEMILIKI ALPHA)' if alpha_vs_random > 0 else '❌ KELAH DIBANDING RANDOM'})")
print(f"  Alpha Gross vs IHSG Index        : {alpha_vs_ihsg*100:+.4f}% ({'✅ STRATEGI UNGGUL (MEMILIKI ALPHA)' if alpha_vs_ihsg > 0 else '❌ KELAH DIBANDING IHSG INDEX'})")
print(f"  Outperformance Net vs Random     : {(strat_net - rand_net)*100:+.4f}%")
print(f"  Outperformance Net vs IHSG ETF   : {(strat_net - ihsg_net)*100:+.4f}%")
