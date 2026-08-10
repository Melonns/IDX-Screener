import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

def evaluate_walk_forward(df: pd.DataFrame, feature_cols: list, target_col: str, xgb_params: dict, n_splits: int = 5):
    """
    Run 5-Fold Walk-Forward Cross Validation using XGBoost.
    Instead of SMOTE, we calculate scale_pos_weight dynamically per fold to handle class imbalance.
    """
    # Sort chronologically by date
    # Note: df is multi-index (date, ticker) or single index (date)
    # Ensure it's sorted by date for TimeSeriesSplit
    df = df.sort_index(level=0)
    
    # We can't just split blindly by row because rows on the same day should be in the same fold.
    # Group by unique dates and split the dates.
    unique_dates = df.index.get_level_values(0).unique()
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    for fold, (train_date_idx, test_date_idx) in enumerate(tscv.split(unique_dates)):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        
        # In financial ML, we must purge N days between train and test to prevent target leakage.
        # Since return_n3 is 3 days ahead, we drop the last 3 days of train_dates
        purge_days = 3
        if len(train_dates) > purge_days:
            train_dates = train_dates[:-purge_days]
            
        # Get data
        train_df = df[df.index.get_level_values(0).isin(train_dates)]
        test_df = df[df.index.get_level_values(0).isin(test_dates)]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Calculate scale_pos_weight (Negative Samples / Positive Samples)
        # This replaces SMOTE natively inside XGBoost
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        # Merge params with scale_pos_weight
        params = xgb_params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        params['eval_metric'] = 'auc'
        params['random_state'] = 42
        params['n_jobs'] = -1
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        # ROC AUC
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
        else:
            auc = 0.5
            ap = 0.0
            
        fold_metrics.append({
            'fold': fold + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'auc': auc,
            'ap': ap
        })
        
    # Aggregate results
    auc_list = [m['auc'] for m in fold_metrics]
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)
    
    # Sharpe-like AUC ratio
    sharpe_auc = (mean_auc - 0.5) / (std_auc + 1e-6)
    
    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'sharpe_auc': sharpe_auc,
        'fold_metrics': fold_metrics
    }
