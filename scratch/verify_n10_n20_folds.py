"""
verify_n10_n20_folds.py — Per-fold Walk-Forward Check for N+10 and N+20 Horizons
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

with db._connect() as conn:
    dates_df = pd.read_sql_query(f"""
        SELECT DISTINCT date FROM daily_prices WHERE is_valid=1 AND date <= '{TRAINING_END}' ORDER BY date ASC
    """, conn)

usable_dates = dates_df['date'].tolist()[60:]
fold_size = len(usable_dates) // 5
fold_ranges = []
for i in range(5):
    s = usable_dates[i * fold_size]
    e = usable_dates[(i+1)*fold_size - 1] if i < 4 else usable_dates[-1]
    fold_ranges.append((s, e))

print("="*75)
print("  PER-FOLD WALK-FORWARD CHECK: N+10 vs N+20 (All Regimes, Training Set)")
print("="*75)

for h in [10, 20]:
    print(f"\n--- HORIZON N+{h} ---")
    fold_nets = []
    
    for fold_idx, (s_date, e_date) in enumerate(fold_ranges, 1):
        fold_rets = []
        for ticker in tickers:
            df = db.get_prices_with_context(ticker, start=s_date, end='2026-03-15')
            if df.empty: continue
            df.index = pd.to_datetime(df.index)
            f_dates = df[(df.index >= pd.Timestamp(s_date)) & (df.index <= pd.Timestamp(e_date))].index
            
            for dt in f_dates:
                row = df.loc[dt]
                if pd.isna(row.get('turnover_5d')) or row['turnover_5d'] < MIN_TURNOVER: continue
                if pd.isna(row.get('vol_accum_5d_rank')) or row['vol_accum_5d_rank'] > RANK_THRESH: continue
                
                entry_p = float(row['Close'])
                df_after = df[df.index > dt].sort_index()
                if len(df_after) >= h:
                    exit_p = float(df_after['Close'].iloc[h - 1])
                    fold_rets.append((exit_p - entry_p) / entry_p)
                    
        n = len(fold_rets)
        g_ev = np.mean(fold_rets) if fold_rets else 0
        n_ev = g_ev - IDX_ROUNDTRIP_COST
        fold_nets.append(n_ev)
        print(f"  Fold {fold_idx} ({s_date} to {e_date}): N={n:3d} | Gross={g_ev*100:+.3f}% | Net (0.4%)={n_ev*100:+.3f}%")
        
    n_pos = sum(1 for ev in fold_nets if ev > 0)
    print(f"  Summary N+{h}: Fold EV Net Positif = {n_pos}/5 | Mean Net EV = {np.mean(fold_nets)*100:+.4f}%")
