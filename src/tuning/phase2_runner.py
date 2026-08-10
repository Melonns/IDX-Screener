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

from data.ml_dataset import build_dataset
from ml.pipeline import evaluate_walk_forward
import config as app_config

def objective(trial, df, feature_cols, target_col):
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    
    results = evaluate_walk_forward(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        xgb_params=xgb_params,
        n_splits=5
    )
    
    # We maximize Sharpe-like AUC (Mean AUC / Std AUC)
    # A high Sharpe AUC means the model is good AND consistent across time folds
    return results['sharpe_auc']

def main():
    parser = argparse.ArgumentParser(description="Run Optuna Phase 2 (ML XGBoost) Tuning")
    parser.add_argument('--trials', type=int, default=50, help='Jumlah trials')
    args = parser.parse_args()
    
    # Load dataset
    print("Membangun dataset ML...")
    df, feature_cols, target_col = build_dataset()
    
    storage_path = os.path.join(app_config.DATA_DIR, 'optuna_ml.db')
    storage = f"sqlite:///{storage_path}"
    study_name = f"phase2_ml_tuning_{datetime.now().strftime('%Y%m%d')}"
    
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
    
    study.optimize(lambda trial: objective(trial, df, feature_cols, target_col), n_trials=args.trials)
    
    print("=" * 50)
    print("Best Trial:")
    best = study.best_trial
    print(f"  Sharpe AUC: {best.value:.4f}")
    print("  Params:")
    for key, val in best.params.items():
        print(f"    {key}: {val}")
        
    print("\nUntuk melihat visualisasi:")
    print(f"optuna-dashboard sqlite:///{storage_path}")

if __name__ == '__main__':
    main()
