import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr
from pathlib import Path

# Fix paths
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / 'src'))

from data.database import DatabaseManager

def get_ihsg_data(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching ^JKSE data from {start_date} to {end_date}...")
    ihsg = yf.download('^JKSE', start=start_date, end=end_date, progress=False)
    # yfinance might return multi-index columns if downloading multiple tickers, 
    # but for single ticker it's usually single index. Let's make sure.
    if isinstance(ihsg.columns, pd.MultiIndex):
        ihsg.columns = ihsg.columns.get_level_values(0)
    
    ihsg.index = ihsg.index.tz_localize(None)
    # Calculate returns
    ihsg['ihsg_ret_5d'] = ihsg['Close'].pct_change(periods=5)
    ihsg['ihsg_ret_20d'] = ihsg['Close'].pct_change(periods=20)
    
    # Calculate MA20 for Market Regime later
    ihsg['ihsg_ma20'] = ihsg['Close'].rolling(window=20).mean()
    ihsg['ihsg_slope_20d'] = (ihsg['ihsg_ma20'] - ihsg['ihsg_ma20'].shift(5)) / ihsg['ihsg_ma20'].shift(5)
    
    # Reset index to make 'Date' a column and string
    ihsg.reset_index(inplace=True)
    # yfinance column is usually 'Date'
    date_col = 'Date' if 'Date' in ihsg.columns else 'index'
    ihsg['date'] = ihsg[date_col].dt.strftime('%Y-%m-%d')
    return ihsg[['date', 'ihsg_ret_5d', 'ihsg_ret_20d', 'ihsg_slope_20d']].dropna()

def calc_cs_ic(df_data, feature_col, return_col):
    daily_ics = []
    for d, group in df_data.groupby('date'):
        if len(group) > 5:
            valid_group = group.dropna(subset=[feature_col, return_col])
            if len(valid_group) > 5:
                ic, _ = spearmanr(valid_group[feature_col], valid_group[return_col])
                if not np.isnan(ic):
                    daily_ics.append(ic)
    
    if not daily_ics:
        return 0.0, 0.0, 1.0
        
    mean_ic = np.mean(daily_ics)
    
    # NEWEY-WEST ADJUSTMENT FOR AUTOCORRELATION (Lag=5 for 5-day window)
    try:
        import statsmodels.api as sm
        # Regress the daily ICs on a constant (vector of ones)
        # to test if the mean is significantly different from 0
        y = np.array(daily_ics)
        X = np.ones(len(y))
        model = sm.OLS(y, X)
        # Use HAC (Heteroskedasticity and Autocorrelation Consistent) covariance matrix
        # maxlags=5 because features are 5-day rolling windows
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        t_stat = results.tvalues[0]
        p_val = results.pvalues[0]
    except ImportError:
        # Fallback to standard t-test if statsmodels is not installed (it should be)
        from scipy.stats import ttest_1samp
        t_stat, p_val = ttest_1samp(daily_ics, 0.0)
        
    return mean_ic, t_stat, p_val

def main():
    db = DatabaseManager('data/idx_screener.db')
    tickers = db.get_tickers()
    
    with db._connect() as conn:
        dates_res = conn.execute("SELECT MIN(date), MAX(date) FROM daily_prices").fetchone()
        min_date, max_date = dates_res
    
    start_fetch = pd.to_datetime(min_date) - pd.DateOffset(days=50)
    end_fetch = pd.to_datetime(max_date) + pd.DateOffset(days=5)
    
    ihsg_df = get_ihsg_data(start_fetch.strftime('%Y-%m-%d'), end_fetch.strftime('%Y-%m-%d'))
    
    all_data = []
    print("Memproses fitur untuk semua emiten...")
    for ticker in tickers:
        df = db.get_prices_with_indicators(ticker)
        if df.empty:
            continue
            
        df = df.copy()
        df.reset_index(inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['ticker'] = ticker
        
        df = pd.merge(df, ihsg_df, on='date', how='left')
        
        df['ret_5d'] = df['Close'].pct_change(periods=5)
        df['rel_strength_5d'] = df['ret_5d'] - df['ihsg_ret_5d']
        
        df['ret_20d'] = df['Close'].pct_change(periods=20)
        df['rel_strength_20d'] = df['ret_20d'] - df['ihsg_ret_20d']
        
        df['money_flow_mult'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
        df['money_flow_vol'] = df['money_flow_mult'] * df['Volume']
        df['vol_accum_5d'] = df['money_flow_vol'].rolling(5).sum() / (df['Volume'].rolling(5).sum() + 1e-8)
        
        df['range_daily'] = df['High'] - df['Low']
        df['volatility_ratio'] = df['range_daily'] / (df['atr_14'] + 1e-8)
        
        df['exit_n3'] = df['Close'].shift(-3)
        df['return_n3'] = (df['exit_n3'] - df['Close']) / df['Close']
        
        df_valid = df.dropna(subset=['return_n3', 'rel_strength_5d', 'vol_accum_5d', 'volatility_ratio'])
        all_data.append(df_valid[['date', 'ticker', 'return_n3', 'rel_strength_5d', 'rel_strength_20d', 'vol_accum_5d', 'volatility_ratio', 'ihsg_slope_20d']])
        
    full_df_master = pd.concat(all_data, ignore_index=True)
    
    cutoffs = ['2026-03-09', '2026-02-09', '2026-01-09']
    features = ['rel_strength_5d', 'vol_accum_5d']
    
    for cutoff in cutoffs:
        print(f"\n{'='*60}")
        print(f"CUTOFF HOLDOUT SENSITIVITY TEST: {cutoff}")
        print(f"{'='*60}")
        full_df = full_df_master[full_df_master['date'] <= cutoff]
        
        uptrend_df = full_df[full_df['ihsg_slope_20d'] > 0]
        downtrend_df = full_df[full_df['ihsg_slope_20d'] <= 0]
        
        print(f"Total Baris (N) : {len(full_df)}")
        print(f"N Uptrend       : {len(uptrend_df)} ({len(uptrend_df)/len(full_df):.1%})")
        print(f"N Downtrend     : {len(downtrend_df)} ({len(downtrend_df)/len(full_df):.1%})")
        print("-" * 60)
        
        for feat in features:
            ic_all, t_all, p_all = calc_cs_ic(full_df, feat, 'return_n3')
            ic_up, t_up, p_up = calc_cs_ic(uptrend_df, feat, 'return_n3')
            ic_dn, t_dn, p_dn = calc_cs_ic(downtrend_df, feat, 'return_n3')
            
            print(f"Feature: {feat}")
            print(f"  Overall   : IC = {ic_all:+.4f} | t-stat = {t_all:+.2f} | p-val = {p_all:.4f}")
            print(f"  Uptrend   : IC = {ic_up:+.4f} | t-stat = {t_up:+.2f} | p-val = {p_up:.4f}")
            print(f"  Downtrend : IC = {ic_dn:+.4f} | t-stat = {t_dn:+.2f} | p-val = {p_dn:.4f}")
            print()

if __name__ == '__main__':
    main()
