"""
contextual_objective.py — Optuna Objective untuk ContextualEngine V2.

Search Space:
- rank_threshold     : Kuantil berapa persen yang dianggap "oversold ekstrem" (1–20%)
- target_return_pct  : Minimum return target untuk dianggap "menang" (0.5–3.0%)

Constraint tambahan di objective function:
- Penalti kalau sinyal per hari > 5 (tidak realistis untuk portofolio individu)
- Penalti kalau ada fold EV < -2% (robustness guard)
- Wajib semua fold punya data (>= 10 sinyal per fold)

Ini tidak mengizinkan tuning manual — semua pencarian dilakukan di dalam
Train/Validation split yang ketat (max_date='2026-02-09').
Holdout (Feb–Agustus 2026) TIDAK disentuh sampai gate check final.
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

HOLDOUT_CUTOFF = '2026-02-09'  # Dikunci mati — jangan diubah tanpa persetujuan
MAX_SIGNALS_PER_DAY = 5        # Batas realistis untuk portofolio individu
N_TRADING_DAYS = 640           # ~3 tahun data training (~640 hari trading)


def contextual_objective(trial: optuna.Trial, db_path: str) -> float:
    # ── Search Space ────────────────────────────────────────────────────────────
    rank_threshold = trial.suggest_float('rank_threshold', 1.0, 20.0)
    target_return_pct = trial.suggest_float('target_return_pct', 0.5, 3.0)

    # ── Setup ────────────────────────────────────────────────────────────────────
    db = DatabaseManager(db_path)
    engine = ContextualEngine(rank_threshold=rank_threshold)

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

    # ── Extract Metrics ──────────────────────────────────────────────────────────
    fold_evs = [f.expected_value('n3') for f in result.folds]
    valid_evs = [ev for ev in fold_evs if ev is not None]

    # Semua fold harus punya data cukup (>= 10 sinyal BULLISH)
    if len(valid_evs) < len(result.folds):
        return -999.0

    # ── Constraint 1: Cek frekuensi sinyal (anti-overfitting frekuensi tinggi) ──
    total_signals = sum(len(f.bullish_signals) for f in result.folds)
    avg_signals_per_day = total_signals / N_TRADING_DAYS
    if avg_signals_per_day > MAX_SIGNALS_PER_DAY:
        # Penalti proporsional, bukan hard cutoff, biar Optuna bisa "belajar"
        penalty = (avg_signals_per_day - MAX_SIGNALS_PER_DAY) * 0.001
    else:
        penalty = 0.0

    # ── Constraint 2: Robustness Guard — fold terburuk tidak boleh kolaps ───────
    min_ev = min(valid_evs)
    if min_ev < -0.02:  # Fold terburuk -2% atau lebih: penalti berat
        return min_ev  # Langsung kembalikan fold terburuk agar Optuna tahu ini buruk

    # ── Objective: Rata-rata EV + bonus konsistensi – penalti frekuensi ─────────
    avg_ev = sum(valid_evs) / len(valid_evs)
    n_positive_folds = sum(1 for ev in valid_evs if ev > 0)

    # Bonus konsistensi: lebih banyak fold positif lebih baik
    consistency_bonus = (n_positive_folds / len(valid_evs)) * 0.001

    return avg_ev + (0.3 * min_ev) + consistency_bonus - penalty
