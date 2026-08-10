"""
Robustness check + breakdown untuk Tahap 2b AND Intersection.
Kemudian bandingkan semua tiga tahap.
"""
import sys, os
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import optuna
import config as app_config
from data.database import DatabaseManager
from scoring.contextual_engine import ContextualEngineAND
from backtest.contextual_backtest import ContextualWalkForwardBacktester
from backtest.engine import IDX_ROUNDTRIP_COST

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Robustness Check ─────────────────────────────────────────────────────────
storage = f'sqlite:///{os.path.join(app_config.DATA_DIR, "optuna_2b_and.db")}'
studies = optuna.get_all_study_names(storage)
study   = optuna.load_study(study_name=studies[-1], storage=storage)
best    = study.best_trial

near_threshold = best.value - abs(best.value * 0.1)
near_trials    = [t for t in study.trials if t.value is not None and t.value >= near_threshold]
vals           = sorted([t.value for t in study.trials if t.value is not None and t.value > -999], reverse=True)

print('=== TAHAP 2b ROBUSTNESS CHECK: AND Intersection ===')
print(f'Best value       : {best.value:.5f}')
print(f'Best params:')
print(f'  rank_rel_strength : {best.params["rank_rel_strength"]:.2f}%')
print(f'  rank_vol_accum    : {best.params["rank_vol_accum"]:.2f}%')
print(f'  target_return_pct : {best.params["target_return_pct"]:.2f}%')
print(f'Near threshold   : {near_threshold:.5f} (dalam 10% dari best)')
print(f'Trial di sekitar : {len(near_trials)} dari {len(study.trials)}')
verdict = 'PLATEAU (robust)' if len(near_trials) >= 5 else 'SPIKE (mencurigakan)'
print(f'Verdict          : {verdict}')
print(f'\nTop 10 objective values:')
for i, v in enumerate(vals[:10]):
    print(f'  {i+1:2}. {v:+.5f}')
print(f'\nParameter cluster di near-best:')
for t in sorted(near_trials, key=lambda x: x.value, reverse=True)[:8]:
    p = t.params
    print(f'  rs={p["rank_rel_strength"]:.1f}%  vol={p["rank_vol_accum"]:.1f}%  '
          f'tgt={p["target_return_pct"]:.2f}%  obj={t.value:+.5f}')

# ── Breakdown Gross EV ────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'=== BREAKDOWN: AND Intersection @ Best Params ===')
db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))
engine = ContextualEngineAND(
    rank_rel_strength=best.params['rank_rel_strength'],
    rank_vol_accum=best.params['rank_vol_accum'],
)
backtester = ContextualWalkForwardBacktester(
    db=db, engine=engine, n_folds=5,
    max_date='2026-02-09',
    target_threshold_pct=best.params['target_return_pct'],
)
result = backtester.run(db.get_tickers())

all_bullish = [s for fold in result.folds for s in fold.bullish_signals if s.return_n3 is not None]
returns     = [s.return_n3 for s in all_bullish]
gross_ev    = np.mean(returns) if returns else 0
net_ev      = gross_ev - IDX_ROUNDTRIP_COST

print(f'\n  Total sinyal BULLISH (5 fold) : {len(all_bullish)}')
print(f'  Rata-rata per hari           : {len(all_bullish)/640:.2f}/hari')
print(f'  Gross EV (sebelum fee)       : {gross_ev*100:+.4f}%')
print(f'  Fee roundtrip                : -{IDX_ROUNDTRIP_COST*100:.2f}%')
print(f'  Net EV (sesudah fee)         : {net_ev*100:+.4f}%')
print(f'  Rasio fee/gross              : {IDX_ROUNDTRIP_COST/abs(gross_ev):.1f}x' if gross_ev != 0 else '')
print(f'\n  Per Fold:')
for fold in result.folds:
    sigs = [s for s in fold.bullish_signals if s.return_n3 is not None]
    if not sigs: continue
    rets  = [s.return_n3 for s in sigs]
    gross = np.mean(rets)
    net   = gross - IDX_ROUNDTRIP_COST
    print(f'  Fold {fold.fold_idx}: N={len(sigs):3d} | Gross={gross*100:+.3f}% | Net={net*100:+.3f}%')

print(f'\n  Fee Sensitivity:')
for label, fee in [('Konservatif 0.40%', 0.004), ('Moderat 0.30%', 0.003), ('Agresif 0.20%', 0.002)]:
    n = gross_ev - fee
    print(f'  {label}: Net = {n*100:+.4f}% ({"✓" if n > 0 else "✗"})')

# ── Summary Semua Tahap ───────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'=== SUMMARY PERBANDINGAN SEMUA TAHAP ===')
print(f'{"Tahap":<30} | {"Best Obj":>9} | {"Gross EV":>9} | {"Net EV":>9} | {"fee/gross":>10}')
print(f'{"-"*65}')
t1_gross, t2a_gross = 0.000669, 0.002300
print(f'{"1: rel_strength standalone":<30} | {-0.004998:>+9.5f} | {t1_gross*100:>+8.4f}% | {(t1_gross-0.004)*100:>+8.4f}% | {"6.0x":>10}')
print(f'{"2a: vol_accum standalone":<30} | {-0.004092:>+9.5f} | {t2a_gross*100:>+8.4f}% | {(t2a_gross-0.004)*100:>+8.4f}% | {"1.7x":>10}')
fee_ratio = f'{IDX_ROUNDTRIP_COST/abs(gross_ev):.1f}x' if gross_ev != 0 else 'N/A'
print(f'{"2b: AND intersection":<30} | {best.value:>+9.5f} | {gross_ev*100:>+8.4f}% | {net_ev*100:>+8.4f}% | {fee_ratio:>10}')
