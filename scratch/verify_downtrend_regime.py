"""
verify_downtrend_regime.py — Verifikasi Konsistensi 5-Fold untuk Regime Downtrend + vol_accum_5d

Tujuan: Memastikan Net EV > 0% pada Downtrend IHSG tidak hanya terjadi di 1 fold,
tetapi konsisten di mayoritas fold (Fold Consistency Check).
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

engine = ContextualEngine(rank_threshold=14.9, feature_mode='vol_accum')
backtester = ContextualWalkForwardBacktester(
    db=db, engine=engine, n_folds=5, max_date=TRAINING_END, target_threshold_pct=2.0
)
res = backtester.run(tickers)

records = []
for f in res.folds:
    for s in f.bullish_signals:
        if s.return_n3 is None: continue
        records.append({
            'ticker': s.ticker,
            'date': s.date,
            'entry_price': s.entry_price,
            'return_n3': s.return_n3,
            'fold': f.fold_idx
        })

df_sigs = pd.DataFrame(records)

with db._connect() as conn:
    market = pd.read_sql_query(f"""
        SELECT date, slope_20d AS ihsg_slope_20d
        FROM market_index
        WHERE date <= '{TRAINING_END}'
    """, conn)

df_sigs = pd.merge(df_sigs, market, on='date', how='left')

# Filter Downtrend (slope < 0)
downtrend_sigs = df_sigs[df_sigs['ihsg_slope_20d'] < 0]

print("\n" + "="*70)
print("  PER-FOLD BREAKDOWN: vol_accum_5d (rank=14.9%) saat IHSG DOWNTREND (slope < 0)")
print("="*70)

fold_evs = []
for fold_id in range(1, 6):
    f_df = downtrend_sigs[downtrend_sigs['fold'] == fold_id]
    n = len(f_df)
    if n == 0:
        print(f"  Fold {fold_id}: No signals")
        continue
    rets = f_df['return_n3'].values
    gross = np.mean(rets)
    net = gross - IDX_ROUNDTRIP_COST
    wr = np.mean(rets >= 0.02)
    fold_evs.append(net)
    print(f"  Fold {fold_id}: N={n:3d} | Gross={gross*100:+.3f}% | Net (0.4%)={net*100:+.3f}% | WR={wr*100:.1f}%")

print("\n" + "="*70)
print("  SUMMARY DOWNTREND REGIME STRATEGY")
print("="*70)
total_gross = downtrend_sigs['return_n3'].mean()
total_net = total_gross - IDX_ROUNDTRIP_COST
n_pos_folds = sum(1 for ev in fold_evs if ev > 0)

print(f"  Total Sinyal           : {len(downtrend_sigs)}")
print(f"  Rata-rata per hari     : {len(downtrend_sigs)/591:.2f} sinyal/hari")
print(f"  Gross EV               : {total_gross*100:+.4f}%")
print(f"  Fee Roundtrip          : -0.4000%")
print(f"  Net EV (0.40%)         : {total_net*100:+.4f}%")
print(f"  Fold EV Positif        : {n_pos_folds} / {len(fold_evs)}")
print(f"  Fold Consistency       : {'✅ PASS' if n_pos_folds >= 3 else '❌ FAIL'}")
