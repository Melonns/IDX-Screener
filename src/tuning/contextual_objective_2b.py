"""
contextual_objective_2b.py — Optuna Objective untuk Tahap 2b: AND Intersection.

Desain berdasarkan sanity check:
- Search space: rank_threshold 10–25% (keduanya, bisa berbeda)
- Constraint MINIMUM sample: >= 30 sinyal per fold (= 150 total) — eksplisit
- Constraint MAXIMUM sinyal: <= 5/hari (realisme portofolio individu)
- Logic: sinyal BULLISH hanya jika KEDUANYA rel_strength_rank DAN vol_accum_rank <= threshold

Holdout 2026-02-09 LOCKED.
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
from scoring.contextual_engine import ContextualEngineAND
from backtest.contextual_backtest import ContextualWalkForwardBacktester

HOLDOUT_CUTOFF   = '2026-02-09'
N_TRADING_DAYS   = 640
MIN_SIGNALS_FOLD = 30    # Eksplisit: per fold minimum buat hasil reliable
MAX_SIGNALS_DAY  = 5     # Eksplisit: max realistis portofolio individu per hari


def objective_2b(trial: optuna.Trial, db_path: str) -> float:
    # ── Search Space ─────────────────────────────────────────────────────────
    # Range 10-25% based on sanity check: semua titik reliable (175-645/fold)
    # Dua threshold bisa berbeda — biar Optuna bebas eksplor asimetri
    rank_rs  = trial.suggest_float('rank_rel_strength', 10.0, 25.0)
    rank_vol = trial.suggest_float('rank_vol_accum',    10.0, 25.0)
    target_return_pct = trial.suggest_float('target_return_pct', 0.5, 3.0)

    # ── Setup ─────────────────────────────────────────────────────────────────
    db = DatabaseManager(db_path)
    engine = ContextualEngineAND(
        rank_rel_strength=rank_rs,
        rank_vol_accum=rank_vol,
    )

    backtester = ContextualWalkForwardBacktester(
        db=db, engine=engine, n_folds=5,
        max_date=HOLDOUT_CUTOFF,
        target_threshold_pct=target_return_pct,
    )

    tickers = db.get_tickers()
    if not tickers:
        return -999.0

    with contextlib.redirect_stdout(None):
        result = backtester.run(tickers)

    # ── Constraint: Minimum Sample per Fold (EKSPLISIT) ───────────────────────
    for fold in result.folds:
        n_bullish = len(fold.bullish_signals)
        if n_bullish < MIN_SIGNALS_FOLD:
            # Terlalu sedikit sinyal — tidak reliable, penalti keras
            return -999.0

    # ── Extract EV per fold ────────────────────────────────────────────────────
    fold_evs = [f.expected_value('n3') for f in result.folds]
    valid_evs = [ev for ev in fold_evs if ev is not None]

    if len(valid_evs) < len(result.folds):
        return -999.0

    # ── Constraint: Maximum Signal Frequency (EKSPLISIT) ─────────────────────
    total_signals = sum(len(f.bullish_signals) for f in result.folds)
    avg_per_day   = total_signals / N_TRADING_DAYS

    if avg_per_day > MAX_SIGNALS_DAY:
        # Penalti proporsional (bukan hard cutoff) biar Optuna bisa belajar
        freq_penalty = (avg_per_day - MAX_SIGNALS_DAY) * 0.002
    else:
        freq_penalty = 0.0

    # ── Robustness Guard: Fold terburuk tidak boleh kolaps ────────────────────
    min_ev = min(valid_evs)
    if min_ev < -0.02:
        return min_ev  # Langsung kembalikan fold terburuk

    # ── Objective Function ─────────────────────────────────────────────────────
    avg_ev = sum(valid_evs) / len(valid_evs)
    n_pos  = sum(1 for ev in valid_evs if ev > 0)
    consistency_bonus = (n_pos / len(valid_evs)) * 0.002  # Lebih besar dari sebelumnya

    return avg_ev + (0.3 * min_ev) + consistency_bonus - freq_penalty
