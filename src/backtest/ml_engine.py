import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from pathlib import Path

_HERE = Path(__file__).parent
_SRC = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.ml_dataset import build_dataset
from backtest.engine import BacktestResult, FoldResult, SignalRecord
from backtest.report import BacktestReporter
import config as app_config

IDX_ROUNDTRIP_COST = 0.004

def run_ml_backtest(threshold_proba: float = 0.70, n_splits: int = 5):
    print(f"\n{'='*65}")
    print(f"  IDX-Screener v2 — ML Trading Backtester (Walk-Forward)")
    print(f"  Threshold Prob: {threshold_proba*100:.0f}%")
    print(f"  Folds         : {n_splits}")
    print(f"{'='*65}")
    
    df, feature_cols, target_col = build_dataset()
    df = df.sort_index(level=0)
    
    unique_dates = df.index.get_level_values(0).unique()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Best params from tuning
    xgb_params = {
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
    
    backtest_res = BacktestResult(
        tickers=df['ticker'].unique().tolist(),
        n_folds=n_splits,
        target_threshold_pct=0.0 # Not used for EV calculation here, EV calculates based on actual returns
    )
    # Note: Phase 1 BacktestResult expected a strict target hit (e.g. 2%).
    # We will pass a dummy 0.0 so we just calculate EV off absolute positive returns vs negative returns.
    
    fold_idx = 1
    all_signals = []
    
    for train_date_idx, test_date_idx in tscv.split(unique_dates):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        
        # Purge to prevent data leakage (3 days)
        purge_days = 3
        if len(train_dates) > purge_days:
            train_dates = train_dates[:-purge_days]
            
        train_df = df[df.index.get_level_values(0).isin(train_dates)]
        test_df = df[df.index.get_level_values(0).isin(test_dates)]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        params = xgb_params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_df = test_df.copy()
        test_df['pred_proba'] = y_pred_proba
        
        # Extract bullish signals
        bullish_df = test_df[test_df['pred_proba'] >= threshold_proba]
        
        fold_res = FoldResult(
            fold_idx=fold_idx,
            train_start=str(train_dates[0].date()) if len(train_dates) > 0 else "",
            train_end=str(train_dates[-1].date()) if len(train_dates) > 0 else "",
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date())
        )
        
        for idx, row in bullish_df.iterrows():
            date_val = idx.date() if isinstance(idx, pd.Timestamp) else str(idx)
            ret_n3 = row['return_n3']
            
            sig = SignalRecord(
                ticker=row['ticker'],
                date=str(date_val),
                score=int(row['pred_proba'] * 100),
                signal='BULLISH',
                entry_price=row['Close'],
                exit_price_n1=None,
                exit_price_n3=row['Close'] * (1 + ret_n3), # proxy
                exit_price_n5=None,
                return_n1=None,
                return_n3=ret_n3,
                return_n5=None,
                hit_target=ret_n3 >= 0.0,
                hit_target_net=ret_n3 >= IDX_ROUNDTRIP_COST
            )
            fold_res.signals.append(sig)
            all_signals.append(sig)
            
        backtest_res.folds.append(fold_res)
        
        print(f"  Fold {fold_idx}: {len(fold_res.signals)} Bullish Signals")
        ev = fold_res.expected_value('n3')
        if ev is not None:
            print(f"    EV (net): {ev*100:+.2f}%")
        else:
            print(f"    EV: N/A (< 10 signals)")
            
        fold_idx += 1
        
    return backtest_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML Trading Backtester")
    parser.add_argument('--threshold', type=float, default=0.70, help='Probability threshold untuk sinyal Bullish')
    parser.add_argument('--folds', type=int, default=5, help='Jumlah fold walk-forward')
    parser.add_argument('--report', type=str, default=None, help='Simpan HTML report')
    args = parser.parse_args()
    
    result = run_ml_backtest(threshold_proba=args.threshold, n_splits=args.folds)
    
    # Calculate score buckets for ML
    buckets = {
        '50-69%': [s for fold in result.folds for s in fold.signals if 50 <= s.score <= 69],
        '70-79%': [s for fold in result.folds for s in fold.signals if 70 <= s.score <= 79],
        '80-89%': [s for fold in result.folds for s in fold.signals if 80 <= s.score <= 89],
        '90-100%': [s for fold in result.folds for s in fold.signals if 90 <= s.score <= 100],
    }
    bucket_stats = {}
    for label, signals in buckets.items():
        n = len(signals)
        if n == 0:
            bucket_stats[label] = {'n': 0, 'note': 'Tidak ada sinyal'}
            continue
            
        returns = [s.return_n3 for s in signals if s.return_n3 is not None]
        wins = [r for r in returns if r >= 0]
        losses = [r for r in returns if r < 0]
        wr = len(wins) / n
        avg_w = float(np.mean(wins)) if wins else 0
        avg_l = float(np.mean(losses)) if losses else 0
        ev_net = (wr * avg_w + (1-wr) * avg_l) - IDX_ROUNDTRIP_COST
        
        bucket_stats[label] = {
            'n': n,
            'win_rate': round(wr, 4),
            'avg_win': round(avg_w, 4),
            'avg_loss': round(avg_l, 4),
            'ev_net': round(ev_net, 4),
            'reliability': 'OK' if n >= 30 else 'HATI-HATI'
        }
    result.score_buckets = bucket_stats
    
    reporter = BacktestReporter(result)
    reporter.print_terminal()
    
    if args.report:
        reporter.save_html(args.report)
        print(f"Report disimpan ke {args.report}")
