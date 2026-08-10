import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

_HERE = Path(__file__).parent
_SRC = _HERE.parent
sys.path.insert(0, str(_SRC))

from data.database import DatabaseManager
from data.index_manager import IndexManager

def compute_contextual_features(db: DatabaseManager):
    print("Fetching index data...")
    mgr = IndexManager(db)
    ihsg = mgr.get_index_data('^JKSE')
    if ihsg.empty:
        print("IHSG data is empty. Run index_manager.py first.")
        return
        
    ihsg['date'] = pd.to_datetime(ihsg['date'])
    ihsg = ihsg.set_index('date')
    
    print("Fetching daily prices for all tickers...")
    tickers = db.get_tickers()
    all_data = []
    
    for ticker in tqdm(tickers, desc="Loading data"):
        # We need raw prices, not indicators
        df = db.get_prices(ticker, valid_only=True)
        if df.empty:
            continue
            
        df = df.copy()
        df.reset_index(inplace=True) # date is column now
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate individual 5d return and volume metrics
        df['ret_5d'] = df['Close'].pct_change(periods=5)
        
        # Turnover
        df['turnover_daily'] = df['Close'] * df['Volume']
        df['turnover_5d'] = df['turnover_daily'].rolling(5).mean()
        
        # Volume accumulation (Money Flow style)
        df['money_flow_mult'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
        df['money_flow_vol'] = df['money_flow_mult'] * df['Volume']
        df['vol_accum_5d'] = df['money_flow_vol'].rolling(5).sum() / (df['Volume'].rolling(5).sum() + 1e-8)
        
        df['ticker'] = ticker
        
        all_data.append(df[['date', 'ticker', 'ret_5d', 'turnover_5d', 'vol_accum_5d']])
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Merge with IHSG to compute relative strength
    print("Computing cross-sectional features...")
    ihsg_ret = ihsg[['ret_5d']].rename(columns={'ret_5d': 'ihsg_ret_5d'}).reset_index()
    full_df = pd.merge(full_df, ihsg_ret, on='date', how='left')
    
    full_df['rel_strength_5d'] = full_df['ret_5d'] - full_df['ihsg_ret_5d']
    
    # Cross-sectional ranking per day
    # Rank 0-100 (pct=True gives 0.0 to 1.0, multiply by 100)
    full_df['rel_strength_5d_rank'] = full_df.groupby('date')['rel_strength_5d'].rank(pct=True, ascending=True) * 100
    full_df['vol_accum_5d_rank'] = full_df.groupby('date')['vol_accum_5d'].rank(pct=True, ascending=True) * 100
    
    # Drop NaNs before saving
    final_df = full_df.dropna(subset=['rel_strength_5d', 'vol_accum_5d']).copy()
    
    print("Saving to database...")
    # Group by ticker and save
    total_inserted = 0
    for ticker, group in tqdm(final_df.groupby('ticker'), desc="Saving"):
        group = group.set_index('date')
        inserted = db.save_contextual_indicators(ticker, group)
        total_inserted += inserted
        
    print(f"Done. Saved {total_inserted} rows of contextual features.")

if __name__ == '__main__':
    db = DatabaseManager(str(_SRC.parent / 'data' / 'idx_screener.db'))
    compute_contextual_features(db)
