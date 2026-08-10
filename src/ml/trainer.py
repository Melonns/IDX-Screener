import os
import sys
import xgboost as xgb
from pathlib import Path
from datetime import datetime
import pandas as pd

_HERE = Path(__file__).parent
_SRC = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.ml_dataset import build_dataset
import config as app_config

def train_final_model():
    print("Membangun dataset ML...")
    df, feature_cols, target_col = build_dataset()
    
    # Best params from Optuna Trial 0
    params = {
        'max_depth': 3,
        'learning_rate': 0.17254716573280354,
        'n_estimators': 233,
        'subsample': 0.7993292420985183,
        'colsample_bytree': 0.5780093202212182,
        'min_child_weight': 2,
        'reg_alpha': 0.0017073967431528124,
        'reg_lambda': 2.9154431891537547,
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # We train on the entire dataset except the last 3 days (due to target leakage)
    unique_dates = df.index.get_level_values(0).unique().sort_values()
    train_dates = unique_dates[:-3]
    train_df = df[df.index.get_level_values(0).isin(train_dates)]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    params['scale_pos_weight'] = num_neg / num_pos if num_pos > 0 else 1.0
    
    print(f"Training final XGBoost model on {len(X_train)} samples...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    model_dir = os.path.join(app_config.BASE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'xgboost_phase2_final.json')
    model.save_model(model_path)
    
    print(f"Final model saved to {model_path}")
    
    # Test MLEngine
    print("\nTesting MLScoringEngine on the last available day...")
    from scoring.ml_engine import MLScoringEngine
    engine = MLScoringEngine(model_path)
    
    # The scoring engine needs a raw DataFrame with 'ema_9', etc. 
    # Luckily, `df` from `build_dataset` contains all raw indicator columns before they were dropped.
    # Let's get the last day data
    last_day = unique_dates[-1].strftime('%Y-%m-%d')
    results = engine.score_all(df, today=last_day)
    
    print(f"\nResults for {last_day}:")
    # Show 2 random results
    count = 0
    for ticker, res in results.items():
        if count >= 2: break
        print(f"\n{ticker} - Skor: {res['skor_total']}% | {res['sinyal']}")
        for b in res['breakdown']:
            print(f"  {b['indikator']}: {b['nilai']} -> {b['kontribusi']}")
        count += 1
        
if __name__ == '__main__':
    train_final_model()
