"""
breakdown_analysis.py — Analisis gross vs net EV untuk setiap tahap.

Jalankan setelah Optuna selesai untuk mendapatkan breakdown lengkap:
  python scratch/breakdown_analysis.py --mode rel_strength --rank 14.9 --target 2.0
  python scratch/breakdown_analysis.py --mode vol_accum --rank X.X --target X.X
"""
import sys
import argparse
sys.path.insert(0, 'src')
import os
import numpy as np
from data.database import DatabaseManager
from scoring.contextual_engine import ContextualEngine
from backtest.contextual_backtest import ContextualWalkForwardBacktester
from backtest.engine import IDX_ROUNDTRIP_COST
import config as app_config


def run_breakdown(feature_mode: str, rank_threshold: float, target_return_pct: float):
    db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))
    tickers = db.get_tickers()

    engine = ContextualEngine(rank_threshold=rank_threshold, feature_mode=feature_mode)
    backtester = ContextualWalkForwardBacktester(
        db=db, engine=engine, n_folds=5,
        max_date='2026-02-09',
        target_threshold_pct=target_return_pct,
    )
    result = backtester.run(tickers)

    all_bullish = [s for fold in result.folds for s in fold.bullish_signals if s.return_n3 is not None]
    total_days = 640

    returns = [s.return_n3 for s in all_bullish]
    gross_ev = np.mean(returns) if returns else 0
    net_ev = gross_ev - IDX_ROUNDTRIP_COST

    print(f"\n{'='*65}")
    print(f"  BREAKDOWN: feature={feature_mode}, rank={rank_threshold}%, target={target_return_pct}%")
    print(f"{'='*65}")
    print(f"\n  Total sinyal BULLISH (5 fold)   : {len(all_bullish)}")
    print(f"  Rata-rata per hari trading      : {len(all_bullish)/total_days:.2f} sinyal/hari")
    print(f"\n  --- Return Analysis ---")
    print(f"  Gross EV (sebelum fee)          : {gross_ev*100:+.4f}%")
    print(f"  Fee roundtrip (konservatif)     : -{IDX_ROUNDTRIP_COST*100:.2f}%")
    print(f"    [komponen: ~0.15% beli + 0.25% jual (incl. PPh 0.1%)]")
    print(f"  Net EV (sesudah fee)            : {net_ev*100:+.4f}%")
    print(f"  Fee mendominasi (fee > |gross|) : {'YA' if IDX_ROUNDTRIP_COST > abs(gross_ev) else 'TIDAK'}")
    if gross_ev != 0:
        print(f"  Rasio fee/gross                 : {IDX_ROUNDTRIP_COST/abs(gross_ev):.1f}x")

    print(f"\n  --- Per Fold Gross vs Net ---")
    for fold in result.folds:
        sigs = [s for s in fold.bullish_signals if s.return_n3 is not None]
        if not sigs:
            continue
        rets = [s.return_n3 for s in sigs]
        gross = np.mean(rets)
        net = gross - IDX_ROUNDTRIP_COST
        wr = sum(1 for r in rets if r >= target_return_pct/100) / len(rets) * 100
        print(f"  Fold {fold.fold_idx}: N={len(sigs):3d} | Gross={gross*100:+.3f}% | Net={net*100:+.3f}% | WR={wr:.1f}%")

    print(f"\n  --- Breakeven Analysis ---")
    print(f"  Gross EV needed untuk breakeven : +{IDX_ROUNDTRIP_COST*100:.2f}%")
    print(f"  Gross EV achieved               : {gross_ev*100:+.4f}%")
    gap = IDX_ROUNDTRIP_COST - gross_ev
    print(f"  Gap yang harus ditutup          : {gap*100:+.4f}%")
    print(f"  Perlu {gap/max(abs(gross_ev), 1e-6):.1f}x lipat gross EV ini buat breakeven" if gross_ev != 0 else "")

    print(f"\n  --- Fee Sensitivity (untuk referensi) ---")
    for fee_label, fee_val in [("Konservatif 0.40%", 0.004), ("Moderat 0.30%", 0.003), ("Agresif 0.20%", 0.002)]:
        net = gross_ev - fee_val
        print(f"  {fee_label}: Net EV = {net*100:+.4f}% ({'✓ positif' if net > 0 else '✗ negatif'})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='rel_strength', choices=['rel_strength', 'vol_accum'])
    parser.add_argument('--rank', type=float, default=14.9)
    parser.add_argument('--target', type=float, default=2.0)
    args = parser.parse_args()
    run_breakdown(args.mode, args.rank, args.target)
