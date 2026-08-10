"""
holdout_test_final.py — Eksekusi Final Holdout Validation (10 Feb 2026 – 7 Aug 2026)

KRITIS:
File ini menjalankan SATU-SATUNYA pengujian independen terakhir pada Holdout Set
menggunakan Formulasi Terkunci yang telah didokumentasikan di implementation_plan.md:

- Strategy: vol_accum_5d_rank <= 14.9%
- Regime Filter: ihsg_slope_20d < 0 (Natural Zero Crossing)
- Turnover Guard: turnover_5d >= 1,000,000,000 Rupiah
- Fee Roundtrip: 0.40% (0.004)
- Horizon Target: N+3 Trading Days
"""

import sys, os
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.database import DatabaseManager
from backtest.engine import IDX_ROUNDTRIP_COST
import config as app_config

HOLDOUT_START = '2026-02-10'
HOLDOUT_END   = '2026-08-07'
RANK_THRESH   = 14.9
MIN_TURNOVER  = 1_000_000_000
TARGET_PCT    = 2.0  # 2.0% return target

db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))
tickers = db.get_tickers()

print("="*70)
print("  FINAL HOLDOUT VALIDATION — IDX-SCREENER V2")
print(f"  Periode Holdout : {HOLDOUT_START} s.d. {HOLDOUT_END}")
print(f"  Jumlah Tickers  : {len(tickers)}")
print(f"  Formulasi Locked: vol_accum_5d_rank <= {RANK_THRESH}% AND ihsg_slope_20d < 0")
print(f"  Fee Roundtrip   : {IDX_ROUNDTRIP_COST*100:.2f}%")
print("="*70)

# Fetch all holdout price and indicator data
all_signals = []

for ticker in tickers:
    # Fetch data up to slightly beyond holdout end for exit evaluation
    df = db.get_prices_with_context(ticker, start='2025-11-01', end='2026-08-20')
    if df.empty:
        continue
    
    df.index = pd.to_datetime(df.index)
    holdout_dates = df[(df.index >= pd.Timestamp(HOLDOUT_START)) & (df.index <= pd.Timestamp(HOLDOUT_END))].index
    
    for dt in holdout_dates:
        row = df.loc[dt]
        
        # 1. Turnover Guard
        turnover = row.get('turnover_5d', 0)
        if pd.isna(turnover) or turnover < MIN_TURNOVER:
            continue
            
        # 2. Feature Rank Check
        vol_rank = row.get('vol_accum_5d_rank')
        if pd.isna(vol_rank) or vol_rank > RANK_THRESH:
            continue
            
        # 3. Natural Regime Filter Check (slope < 0)
        ihsg_slope = row.get('ihsg_slope_20d')
        if pd.isna(ihsg_slope) or ihsg_slope >= 0:
            continue
            
        # Signal Triggered!
        entry_price = float(row['Close'])
        date_str = dt.strftime('%Y-%m-%d')
        
        # Calculate N+3 exit price
        df_after = df[df.index > dt].sort_index()
        if len(df_after) >= 3:
            exit_price_n3 = float(df_after['Close'].iloc[2])
            ret_n3 = (exit_price_n3 - entry_price) / entry_price
        else:
            ret_n3 = None
            
        all_signals.append({
            'ticker': ticker,
            'date': date_str,
            'entry_price': entry_price,
            'exit_price_n3': exit_price_n3 if ret_n3 is not None else None,
            'return_n3': ret_n3,
            'vol_rank': vol_rank,
            'ihsg_slope': ihsg_slope
        })

df_results = pd.DataFrame(all_signals)
valid_signals = df_results.dropna(subset=['return_n3']).copy()

print("\n" + "="*70)
print("  HASIL KINERJA HOLDOUT SET (FEBRUARI – AGUSTUS 2026)")
print("="*70)

n_total = len(valid_signals)
if n_total == 0:
    print("  ❌ TIDAK ADA SINYAL YANG TERPICU PADA PERIODE HOLDOUT.")
    sys.exit(0)

returns = valid_signals['return_n3'].values
gross_ev = np.mean(returns)
net_ev = gross_ev - IDX_ROUNDTRIP_COST
win_rate = np.mean(returns >= (TARGET_PCT / 100.0))
win_rate_gross_pos = np.mean(returns > 0)
avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0

print(f"  Total Sinyal Bullish Valid : {n_total} sinyal")
print(f"  Rata-rata sinyal per hari  : {n_total / 125:.2f} sinyal/hari (selama 6 bulan)")
print(f"  Gross EV (sebelum fee)     : {gross_ev*100:+.4f}%")
print(f"  Fee Roundtrip              : -{IDX_ROUNDTRIP_COST*100:.2f}%")
print(f"  Net EV (sesudah fee)       : {net_ev*100:+.4f}%")
print(f"  Win Rate (Return >= {TARGET_PCT}%) : {win_rate*100:.2f}%")
print(f"  Win Rate (Return > 0%)    : {win_rate_gross_pos*100:.2f}%")
print(f"  Average Win (Gross)        : {avg_win*100:+.4f}%")
print(f"  Average Loss (Gross)       : {avg_loss*100:+.4f}%")
print(f"  Max Drawdown (Single Trade): {np.min(returns)*100:+.2f}%")

print("\n" + "="*70)
print("  GATE CHECK EVALUATION (HOLDOUT SET)")
print("="*70)

gate_sample = n_total >= 30
gate_net_ev = net_ev > 0.0
gate_wr     = win_rate >= 0.50

print(f"  [{'✅' if gate_sample else '❌'}] Sample Size Guard (N >= 30)      : {n_total} sinyal")
print(f"  [{'✅' if gate_net_ev else '❌'}] Net EV Positif (Net EV > 0.0%)    : {net_ev*100:+.4f}%")
print(f"  [{'✅' if gate_wr else '❌'}] Win Rate Guard (WR >= 50.0%)     : {win_rate*100:.2f}%")

all_passed = gate_sample and gate_net_ev and gate_wr
print(f"\n  KESIMPULAN FINAL HOLDOUT: {'✅ KANDIDAT VALID & PROFITABLE' if all_passed else '❌ GAGAL MEMENUHI GATE HOLDOUT'}")
