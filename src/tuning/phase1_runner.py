import os
import sys
import argparse
import optuna
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).parent
_SRC = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from tuning.phase1_objective import objective
from data.database import DatabaseManager
import config as app_config

def main():
    parser = argparse.ArgumentParser(description="Run Optuna Phase 1 Tuning")
    parser.add_argument('--db', type=str, default=None, help='Path ke SQLite DB')
    parser.add_argument('--trials', type=int, default=100, help='Jumlah trials')
    args = parser.parse_args()
    
    db_path = args.db or os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    
    # Simpan study di sqlite agar bisa di-resume & dicek pakai optuna-dashboard
    storage_path = os.path.join(app_config.DATA_DIR, 'optuna.db')
    storage = f"sqlite:///{storage_path}"
    
    study_name = f"phase1_tuning_{datetime.now().strftime('%Y%m%d')}"
    
    # Gunakan TPESampler (Tree-structured Parzen Estimator)
    sampler = optuna.samplers.TPESampler(seed=42)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True
    )
    
    print(f"Memulai Optuna Study: {study_name}")
    print(f"Target: {args.trials} trials")
    print(f"Storage: {storage_path}")
    print("=" * 50)
    
    # Jalankan optimasi
    study.optimize(lambda trial: objective(trial, db_path), n_trials=args.trials)
    
    print("=" * 50)
    print("Best Trial:")
    best = study.best_trial
    print(f"  Value (Score): {best.value:.4f}")
    print("  Params:")
    for key, val in best.params.items():
        print(f"    {key}: {val}")
        
    print("\nUntuk melihat visualisasi:")
    print(f"optuna-dashboard sqlite:///{storage_path}")

if __name__ == '__main__':
    main()
