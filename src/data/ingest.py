import os
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Add data directory to path so we can import universe.py directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_target_tickers():
    """Membaca daftar saham hasil filter dari liquid_universe.txt atau universe.py."""
    filepath = os.path.join(config.DATA_DIR, 'liquid_universe.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        if tickers:
            return tickers

    try:
        import universe
        print("Mencari universe saham likuid dengan src/data/universe.py...")
        all_tickers = universe.get_all_idx_tickers()
        tickers = universe.filter_by_liquidity(all_tickers)
        if tickers:
            with open(filepath, 'w', encoding='utf-8') as f:
                for t in tickers:
                    f.write(f"{t}\n")
            return tickers
    except Exception as e:
        print(f"Gagal menggunakan universe.py: {e}")

    print("Fallback ke daftar ticker default karena universe filter gagal.")
    return config.DEFAULT_TICKERS

def download_data(tickers=None, period="5y", interval=config.DEFAULT_INTERVAL):
    """
    Download historical data for a list of tickers.
    If period is 5y, it downloads a massive dataset suitable for ML training.
    """
    if tickers is None:
        tickers = get_target_tickers()
        
    print(f"Mengunduh data sejarah panjang ({period}) untuk {len(tickers)} emiten...")
    
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
