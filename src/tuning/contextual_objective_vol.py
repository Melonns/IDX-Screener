"""
contextual_objective_vol.py — Optuna Objective untuk vol_accum_5d standalone.

Tahap 2a: Test vol_accum_5d SENDIRIAN (apple-to-apple dengan rel_strength_5d).
Search space identik dengan Tahap 1 biar hasilnya comparable.
"""

import sys
import optuna
import contextlib
from pathlib import Path

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
from scoring.contextual_engine import ContextualEngine
from backtest.contextual_backtest import ContextualWalkForwardBacktester

HOLDOUT_CUTOFF = '2026-02-09'
MAX_SIGNALS_PER_DAY = 5
N_TRADING_DAYS = 640


def vol_accum_objective(trial: optuna.Trial, db_path: str) -> float:
    rank_threshold = trial.suggest_float('rank_threshold', 1.0, 20.0)
    target_return_pct = trial.suggest_float('target_return_pct', 0.5, 3.0)

    db = DatabaseManager(db_path)
    # Identik dengan Tahap 1, KECUALI feature_mode='vol_accum'
    engine = ContextualEngine(rank_threshold=rank_threshold, feature_mode='vol_accum')

    backtester = ContextualWalkForwardBacktester(
        db=db,
        engine=engine,
        n_folds=5,
        max_date=HOLDOUT_CUTOFF,
        target_threshold_pct=target_return_pct,
    )

    tickers = db.get_tickers()
    if not tickers:
        return -999.0

    with contextlib.redirect_stdout(None):
        result = backtester.run(tickers)

    fold_evs = [f.expected_value('n3') for f in result.folds]
    valid_evs = [ev for ev in fold_evs if ev is not None]

    if len(valid_evs) < len(result.folds):
        return -999.0

    total_signals = sum(len(f.bullish_signals) for f in result.folds)
    avg_signals_per_day = total_signals / N_TRADING_DAYS
    if avg_signals_per_day > MAX_SIGNALS_PER_DAY:
        penalty = (avg_signals_per_day - MAX_SIGNALS_PER_DAY) * 0.001
    else:
        penalty = 0.0

    min_ev = min(valid_evs)
    if min_ev < -0.02:
        return min_ev

    avg_ev = sum(valid_evs) / len(valid_evs)
    n_positive_folds = sum(1 for ev in valid_evs if ev > 0)
    consistency_bonus = (n_positive_folds / len(valid_evs)) * 0.001

    return avg_ev + (0.3 * min_ev) + consistency_bonus - penalty
