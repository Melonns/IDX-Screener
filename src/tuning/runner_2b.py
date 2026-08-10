"""
runner_2b.py — Optuna runner untuk Tahap 2b: AND intersection.

Pakai:
  python src/tuning/runner_2b.py --trials 50
"""

import os, sys, time, argparse, optuna
from pathlib import Path
from datetime import datetime, timedelta

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from tuning.contextual_objective_2b import objective_2b
import config as app_config


def make_callback(n_trials: int):
    start = time.time()
    def cb(study, trial):
        done = trial.number + 1
        avg  = (time.time() - start) / done
        eta  = str(timedelta(seconds=int(avg * (n_trials - done))))
        best = study.best_trial
        p    = trial.params
        v    = trial.value if trial.value is not None else float('nan')
        flag = "✓ BEST" if trial.number == best.number else "  "
        print(
            f"[{done:>3}/{n_trials}] {flag} "
            f"rs={p.get('rank_rel_strength',0):.1f}%  "
            f"vol={p.get('rank_vol_accum',0):.1f}%  "
            f"tgt={p.get('target_return_pct',0):.2f}%  "
            f"obj={v:+.5f} | best={best.value:+.5f}  ETA={eta}",
            flush=True
        )
    return cb


def main():
    parser = argparse.ArgumentParser(description="Tahap 2b AND Intersection — Optuna")
    parser.add_argument('--db', type=str, default=None)
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()

    db_path = args.db or os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    storage_path = os.path.join(app_config.DATA_DIR, 'optuna_2b_and.db')
    storage = f"sqlite:///{storage_path}"

    study_name = f"ctx_2b_AND_{datetime.now().strftime('%Y%m%d_%H%M')}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42),
        direction="maximize",
        load_if_exists=True,
    )

    print(f"=== Tahap 2b AND Intersection — Optuna ===", flush=True)
    print(f"Study    : {study_name}", flush=True)
    print(f"Trials   : {args.trials}", flush=True)
    print(f"Storage  : {storage_path}", flush=True)
    print(f"Holdout  : 2026-02-09 (LOCKED)", flush=True)
    print(f"Logic    : BULLISH iff rel_strength_rank <= X AND vol_accum_rank <= Y", flush=True)
    print(f"Constraints:", flush=True)
    print(f"  Min sinyal per fold : 30 (eksplisit)", flush=True)
    print(f"  Max sinyal per hari : 5  (eksplisit)", flush=True)
    print(f"Search space:", flush=True)
    print(f"  rank_rel_strength   : 10–25%", flush=True)
    print(f"  rank_vol_accum      : 10–25%", flush=True)
    print(f"  target_return_pct   : 0.5–3.0%", flush=True)
    print(f"Started  : {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print("=" * 65, flush=True)

    study.optimize(
        lambda trial: objective_2b(trial, db_path),
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
        print(f"    {k}: {v:.4f}", flush=True)
    print(f"\n  optuna-dashboard sqlite:///{storage_path}", flush=True)


if __name__ == '__main__':
    main()
