"""
contextual_runner.py — Optuna runner untuk ContextualEngine V2.

Pakai:
  python src/tuning/contextual_runner.py --trials 50
"""

import os
import sys
import argparse
import optuna
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from tuning.contextual_objective import contextual_objective
import config as app_config


def main():
    parser = argparse.ArgumentParser(description="Contextual Engine V2 — Optuna Tuning")
    parser.add_argument('--db', type=str, default=None)
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()

    db_path = args.db or os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    storage_path = os.path.join(app_config.DATA_DIR, 'optuna_contextual.db')
    storage = f"sqlite:///{storage_path}"

    study_name = f"contextual_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sampler = optuna.samplers.TPESampler(seed=42)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"=== Contextual Engine V2 — Optuna ===")
    print(f"Study   : {study_name}")
    print(f"Trials  : {args.trials}")
    print(f"Storage : {storage_path}")
    print(f"Holdout : 2026-02-09 (LOCKED)")
    print("=" * 40)

    study.optimize(
        lambda trial: contextual_objective(trial, db_path),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print(f"  Score (Objective): {best.value:.6f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"\nLihat hasil lengkap:")
    print(f"  optuna-dashboard sqlite:///{storage_path}")


if __name__ == '__main__':
    main()
