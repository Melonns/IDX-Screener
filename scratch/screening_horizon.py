"""
screening_horizon.py — Pengujian Multi-Horizon (N+1 s.d. N+20 Hari) pada vol_accum_5d

Tujuan: Menguji apakah memperpanjang horizon holding period (N+5, N+10, N+15, N+20)
dapat meningkatkan Gross EV sehingga melampaui biaya transaksi 0.40% roundtrip.

Dataset: Training Set (Agustus 2023 – 9 Februari 2026).
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

db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))
tickers = db.get_tickers()

print("="*75)
print("  MULTI-HORIZON SCREENING: vol_accum_5d (rank <= 14.9%)")
print(f"  Periode Data   : Training Set (Agustus 2023 s.d. {TRAINING_END})")
print(f"  Fee Roundtrip  : {IDX_ROUNDTRIP_COST*100:.2f}%")
print("="*75)

horizons = [1, 3, 5, 10, 15, 20]
horizon_results = {h: [] for h in horizons}

for ticker in tickers:
    df = db.get_prices_with_context(ticker, end='2026-03-15')
    if df.empty or len(df) < 60:
        continue
        
    df.index = pd.to_datetime(df.index)
    train_dates = df[df.index <= pd.Timestamp(TRAINING_END)].index
    
    for dt in train_dates:
        row = df.loc[dt]
        
        # 1. Turnover Guard
        turnover = row.get('turnover_5d', 0)
        if pd.isna(turnover) or turnover < MIN_TURNOVER:
            continue
            
        # 2. Feature Rank Check
        vol_rank = row.get('vol_accum_5d_rank')
        if pd.isna(vol_rank) or vol_rank > RANK_THRESH:
            continue
            
        entry_price = float(row['Close'])
        df_after = df[df.index > dt].sort_index()
        
        # Calculate returns for each horizon
        for h in horizons:
            if len(df_after) >= h:
                exit_p = float(df_after['Close'].iloc[h - 1])
                ret = (exit_p - entry_price) / entry_price
                horizon_results[h].append(ret)

print(f"\n{'Horizon':<10} | {'N Trade':>8} | {'Gross EV':>9} | {'Net EV (0.4%)':>12} | {'Win Rate (>0)':>12} | {'Fee/Gross':>10}")
print("-" * 75)

for h in horizons:
    rets = np.array(horizon_results[h])
    n = len(rets)
    if n == 0: continue
    gross = np.mean(rets)
    net = gross - IDX_ROUNDTRIP_COST
    wr_pos = np.mean(rets > 0) * 100
    ratio_str = f"{IDX_ROUNDTRIP_COST/gross:.1f}x" if gross > 0 else "N/A (Neg)"
    
    net_str = f"{net*100:>+11.4f}%"
    status_flag = "✓ PROFIT" if net > 0 else "✗ LOSS"
    
    print(f"N+{h:<8} | {n:>8,d} | {gross*100:>+8.4f}% | {net_str} | {wr_pos:>11.1f}% | {ratio_str:>10}  {status_flag}")

print("\n" + "="*75)
print("  DOWNTREND REGIME ONLY: MULTI-HORIZON SCREENING (slope < 0)")
print("="*75)

downtrend_results = {h: [] for h in horizons}

for ticker in tickers:
    df = db.get_prices_with_context(ticker, end='2026-03-15')
    if df.empty or len(df) < 60:
        continue
        
    df.index = pd.to_datetime(df.index)
    train_dates = df[df.index <= pd.Timestamp(TRAINING_END)].index
    
    for dt in train_dates:
        row = df.loc[dt]
        
        if pd.isna(row.get('turnover_5d')) or row['turnover_5d'] < MIN_TURNOVER:
            continue
        if pd.isna(row.get('vol_accum_5d_rank')) or row['vol_accum_5d_rank'] > RANK_THRESH:
            continue
        if pd.isna(row.get('ihsg_slope_20d')) or row['ihsg_slope_20d'] >= 0:
            continue
            
        entry_price = float(row['Close'])
        df_after = df[df.index > dt].sort_index()
        
        for h in horizons:
            if len(df_after) >= h:
                exit_p = float(df_after['Close'].iloc[h - 1])
                ret = (exit_p - entry_price) / entry_price
                downtrend_results[h].append(ret)

print(f"\n{'Horizon':<10} | {'N Trade':>8} | {'Gross EV':>9} | {'Net EV (0.4%)':>12} | {'Win Rate (>0)':>12} | {'Fee/Gross':>10}")
print("-" * 75)

for h in horizons:
    rets = np.array(downtrend_results[h])
    n = len(rets)
    if n == 0: continue
    gross = np.mean(rets)
    net = gross - IDX_ROUNDTRIP_COST
    wr_pos = np.mean(rets > 0) * 100
    ratio_str = f"{IDX_ROUNDTRIP_COST/gross:.1f}x" if gross > 0 else "N/A (Neg)"
    
    net_str = f"{net*100:>+11.4f}%"
    status_flag = "✓ PROFIT" if net > 0 else "✗ LOSS"
    
    print(f"N+{h:<8} | {n:>8,d} | {gross*100:>+8.4f}% | {net_str} | {wr_pos:>11.1f}% | {ratio_str:>10}  {status_flag}")
