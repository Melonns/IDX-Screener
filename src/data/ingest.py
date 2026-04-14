import os
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def download_data(tickers=config.DEFAULT_TICKERS, period=config.DEFAULT_PERIOD, interval=config.DEFAULT_INTERVAL):
    """
    Download historical data for a list of tickers.
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    
    # yfinance download works well with a space-separated string of tickers
    tickers_str = " ".join(tickers)
    data = yf.download(tickers_str, period=period, interval=interval, group_by='ticker')
    
    saved_files = []
    
    # Handle single ticker case vs multiple tickers case
    if len(tickers) == 1:
        ticker = tickers[0]
        df = data.copy()
        if not df.empty:
            df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
            filename = f"{ticker}_{period}_{interval}.csv"
            filepath = os.path.join(config.RAW_DATA_DIR, filename)
            df.to_csv(filepath)
            saved_files.append(filepath)
            print(f"Saved {ticker} to {filepath}")
    else:
        for ticker in tickers:
            if ticker in data:
                df = data[ticker].copy()
                if not df.empty:
                    # Drop NA values that might occur due to mismatched trading days
                    df.dropna(how='all', inplace=True)
                    filename = f"{ticker}_{period}_{interval}.csv"
                    filepath = os.path.join(config.RAW_DATA_DIR, filename)
                    df.to_csv(filepath)
                    saved_files.append(filepath)
                    print(f"Saved {ticker}[{len(df)} rows] to {filepath}")
            else:
                print(f"Warning: No data found for {ticker}")
                
    return saved_files

if __name__ == "__main__":
    download_data()
