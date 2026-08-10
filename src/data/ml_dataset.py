import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
import config as app_config

def build_dataset(db_path: str = None, n_forward: int = 3, top_pct: float = 0.3):
    """
    Build ML dataset from SQLite.
    Target: 1 if return N+forward is in the top_pct of cross-sectional rank for that day.
    """
    if db_path is None:
        db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
        
    db = DatabaseManager(db_path)
    
    print("Loading data from database...")
    # Fetch all prices and indicators
    tickers = db.get_tickers()
    all_data = []
    
    for ticker in tickers:
        df = db.get_prices_with_indicators(ticker)
        if df.empty or len(df) < 50:
            continue
            
        df = df.sort_index()
        df['ticker'] = ticker
        
        # Calculate forward return
        df[f'return_n{n_forward}'] = (df['Close'].shift(-n_forward) - df['Close']) / df['Close']
        
        # Calculate derived normalized features for ML
        # Price relative to EMAs
        df['ema_9_dist'] = df['Close'] / df['ema_9'] - 1
        df['ema_21_dist'] = df['Close'] / df['ema_21'] - 1
        df['ema_50_dist'] = df['Close'] / df['ema_50'] - 1
        
        # %B (Bollinger Band position)
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_pct_b'] = np.where(bb_range == 0, 0, (df['Close'] - df['bb_lower']) / bb_range)
        
        all_data.append(df)
        
    if not all_data:
        raise ValueError("No valid data found in database.")
        
    # Combine all tickers
    full_df = pd.concat(all_data)
    
    # Drop rows without future returns
    full_df = full_df.dropna(subset=[f'return_n{n_forward}'])
    
    # Calculate cross-sectional features and targets per date
    print("Calculating Cross-Sectional Ranks...")
    features_to_rank = ['rsi_14', 'macd_diff', 'volume_ratio_20d', 'bb_pct_b', 'bb_width']
    ranked_dfs = []
    
    for date, group in full_df.groupby(level=0):
        # We need at least 10 stocks on a given day to calculate meaningful ranks
        if len(group) < 10:
            continue
            
        group = group.copy()
        
        # Calculate target label: Top 30% of returns
        ret_threshold = group[f'return_n{n_forward}'].quantile(1 - top_pct)
        group['target'] = (group[f'return_n{n_forward}'] >= ret_threshold).astype(int)
        
        # Calculate cross-sectional rank of features (0 to 1)
        for f in features_to_rank:
            group[f'{f}_cs_rank'] = group[f].rank(pct=True)
            
        ranked_dfs.append(group)
        
    final_df = pd.concat(ranked_dfs)
    final_df = final_df.sort_index()
    
    # Prepare X and y
    feature_cols = [
        # Normalized raw features
        'rsi_14', 'macd_diff', 'macd', 'macd_signal', 'volume_ratio_20d',
        'ema_9_dist', 'ema_21_dist', 'ema_50_dist', 'bb_pct_b', 'bb_width',
        # Cross-sectional rank features
        'rsi_14_cs_rank', 'macd_diff_cs_rank', 'volume_ratio_20d_cs_rank',
        'bb_pct_b_cs_rank', 'bb_width_cs_rank'
    ]
    
    # Drop NaNs in features
    final_df = final_df.dropna(subset=feature_cols)
    
    print(f"Dataset ready. Total samples: {len(final_df)}")
    print(f"Target distribution (1=Top {top_pct*100}%, 0=Rest):")
    print(final_df['target'].value_counts(normalize=True))
    
    return final_df, feature_cols, 'target'

if __name__ == '__main__':
    df, features, target = build_dataset()
    print("\nFeatures used:")
    for f in features:
        print(f" - {f}")
