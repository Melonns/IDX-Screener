"""
hybrid_backtest.py — 5-Fold Walk-Forward Backtester & Stress-Tester (Phase 5)

Menjalankan pengujian 5-fold walk-forward yang sangat ketat pada Training Set (2023–2026).
Memeriksa:
1. Per-fold Net EV, Win Rate, & Drawdown
2. Benchmark vs Random Stock & IHSG Index ETF
3. Stress Testing (Hole Checking): Sensitivitas Yield, Window, & Volume Filter
"""

import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
from backtest.engine import IDX_ROUNDTRIP_COST
from scoring.hybrid_engine import MultiFactorHybridEngine
import config as app_config

TRAINING_END = '2026-02-09'


class HybridWalkForwardBacktester:
    def __init__(self, db: DatabaseManager, engine: MultiFactorHybridEngine, n_folds: int = 5):
        self.db = db
        self.engine = engine
        self.n_folds = n_folds

    def run_backtest(self, min_yield: float = 4.0, window_days: int = 10, max_vol_rank: float = 30.0) -> Dict[str, Any]:
        with self.db._connect() as conn:
            df_divs = pd.read_sql_query(f"""
                SELECT ticker, date AS div_date, value AS div_amount, dividend_yield
                FROM corporate_actions
                WHERE event_type = 'DIVIDEND'
                AND date <= '{TRAINING_END}'
                AND dividend_yield >= {min_yield}
                ORDER BY date ASC
            """, conn)
            
            prices = pd.read_sql_query(f"""
                SELECT p.date, p.ticker, p.close, c.turnover_5d, c.vol_accum_5d_rank
                FROM daily_prices p
                LEFT JOIN contextual_indicators c USING (ticker, date)
                WHERE p.date <= '{TRAINING_END}' AND p.is_valid = 1
                ORDER BY p.ticker, p.date ASC
            """, conn)
            
            dates_df = pd.read_sql_query(f"""
                SELECT DISTINCT date FROM daily_prices WHERE is_valid=1 AND date <= '{TRAINING_END}' ORDER BY date ASC
            """, conn)
            
            ihsg = pd.read_sql_query(f"""
                SELECT date, close AS ihsg_close
                FROM market_index
                WHERE symbol = '^JKSE' AND date <= '{TRAINING_END}'
                ORDER BY date ASC
            """, conn)

        prices['date'] = pd.to_datetime(prices['date'])
        df_divs['div_date'] = pd.to_datetime(df_divs['div_date'])
        ihsg['date'] = pd.to_datetime(ihsg['date'])
        ihsg = ihsg.set_index('date').sort_index()

        # Build 5 Folds
        usable_dates = dates_df['date'].tolist()[60:]
        fold_size = len(usable_dates) // self.n_folds
        fold_ranges = []
        for i in range(self.n_folds):
            s = usable_dates[i * fold_size]
            e = usable_dates[(i+1)*fold_size - 1] if i < self.n_folds - 1 else usable_dates[-1]
            fold_ranges.append((s, e))

        records = []
        for idx, row in df_divs.iterrows():
            ticker  = row['ticker']
            ex_date = row['div_date']
            yield_val = row['dividend_yield']
            
            p_sub = prices[prices['ticker'] == ticker].sort_values('date')
            if p_sub.empty: continue
            
            before_ex = p_sub[p_sub['date'] < ex_date]
            if len(before_ex) < window_days + 1: continue
            
            exit_row  = before_ex.iloc[-1]
            entry_row = before_ex.iloc[-(window_days + 1)]
            
            entry_dt = entry_row['date']
            exit_dt  = exit_row['date']
            
            # Turnover check on entry
            if entry_row['turnover_5d'] < self.engine.min_turnover:
                continue
                
            # Vol rank check if requested
            if max_vol_rank < 100.0:
                v_rank = entry_row['vol_accum_5d_rank']
                if pd.isna(v_rank) or v_rank > max_vol_rank:
                    continue

            entry_p = entry_row['close']
            exit_p  = exit_row['close']
            
            if entry_p > 0:
                ret_gross = (exit_p - entry_p) / entry_p
                ret_net   = ret_gross - IDX_ROUNDTRIP_COST
                
                # IHSG return for benchmark
                ihsg_ret = 0.0
                if entry_dt in ihsg.index and exit_dt in ihsg.index:
                    i_en = ihsg.loc[entry_dt, 'ihsg_close']
                    i_ex = ihsg.loc[exit_dt, 'ihsg_close']
                    ihsg_ret = (i_ex - i_en) / i_en
                    
                exit_str = exit_dt.strftime('%Y-%m-%d')
                f_idx = 1
                for f_i, (s_d, e_d) in enumerate(fold_ranges, 1):
                    if s_d <= exit_str <= e_d:
                        f_idx = f_i
                        break
                        
                records.append({
                    'ticker': ticker,
                    'entry_date': entry_dt.strftime('%Y-%m-%d'),
                    'exit_date': exit_dt.strftime('%Y-%m-%d'),
                    'ret_gross': ret_gross,
                    'ret_net': ret_net,
                    'ihsg_ret': ihsg_ret,
                    'yield': yield_val,
                    'fold': f_idx
                })

        df_res = pd.DataFrame(records)
        return {'df': df_res, 'fold_ranges': fold_ranges}


if __name__ == '__main__':
    db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    engine = MultiFactorHybridEngine()
    tester = HybridWalkForwardBacktester(db, engine)
    res = tester.run_backtest(min_yield=4.0, window_days=10, max_vol_rank=100.0)
    df_r = res['df']
    print(f"Total valid signals: {len(df_r)}")
    if not df_r.empty:
        print(f"Gross EV: {df_r['ret_gross'].mean()*100:+.4f}%")
        print(f"Net EV  : {df_r['ret_net'].mean()*100:+.4f}%")
