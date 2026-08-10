"""
vol_accum_runner.py — Optuna runner untuk vol_accum_5d standalone (Tahap 2a).

Pakai:
  python src/tuning/vol_accum_runner.py --trials 50
"""

import os
import sys
import time
import argparse
import optuna
from pathlib import Path
from datetime import datetime, timedelta

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from tuning.contextual_objective_vol import vol_accum_objective
import config as app_config


def make_callback(n_trials: int):
    start_time = time.time()

    def callback(study: optuna.Study, trial: optuna.Trial):
        elapsed = time.time() - start_time
        done = trial.number + 1
        avg_sec = elapsed / done
        eta_sec = avg_sec * (n_trials - done)
        eta_str = str(timedelta(seconds=int(eta_sec)))

        best = study.best_trial
        params = trial.params
        val = trial.value if trial.value is not None else float('nan')

        status = "✓ BEST" if trial.number == best.number else "  "
        print(
            f"[{done:>3}/{n_trials}] {status} "
            f"rank={params.get('rank_threshold', 0):.1f}%  "
            f"target={params.get('target_return_pct', 0):.2f}%  "
            f"obj={val:+.5f}  "
            f"| best={best.value:+.5f}  ETA={eta_str}",
            flush=True
        )

    return callback


def main():
    parser = argparse.ArgumentParser(description="vol_accum_5d Standalone — Optuna (Tahap 2a)")
    parser.add_argument('--db', type=str, default=None)
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()

    db_path = args.db or os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    # Studi terpisah dari Tahap 1 — history bersih
    storage_path = os.path.join(app_config.DATA_DIR, 'optuna_vol_accum.db')
    storage = f"sqlite:///{storage_path}"

    study_name = f"vol_accum_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sampler = optuna.samplers.TPESampler(seed=42)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"=== vol_accum_5d Standalone — Optuna (Tahap 2a) ===", flush=True)
    print(f"Study   : {study_name}", flush=True)
    print(f"Trials  : {args.trials}", flush=True)
    print(f"Storage : {storage_path}", flush=True)
    print(f"Holdout : 2026-02-09 (LOCKED)", flush=True)
    print(f"Feature : vol_accum_5d_rank (standalone, apple-to-apple vs Tahap 1)", flush=True)
    print(f"Started : {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print("=" * 65, flush=True)

    study.optimize(
        lambda trial: vol_accum_objective(trial, db_path),
        n_trials=args.trials,
        callbacks=[make_callback(args.trials)],
        show_progress_bar=False,
    )

    print("\n" + "=" * 65, flush=True)
    print("=== BEST TRIAL ===", flush=True)
    best = study.best_trial
    print(f"  Score (Objective): {best.value:.6f}", flush=True)
    print("  Params:", flush=True)
    for k, v in best.params.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}", flush=True)

    print(f"\nLihat hasil lengkap:", flush=True)
    print(f"  optuna-dashboard sqlite:///{storage_path}", flush=True)


if __name__ == '__main__':
    main()
