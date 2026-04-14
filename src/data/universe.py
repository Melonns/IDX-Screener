import pandas as pd
import yfinance as yf
import os
import sys

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_all_idx_tickers():
    """
    Scrape Wikipedia for all listed companies in IDX.
    """
    print("Mencari daftar emiten IHSG dari Wikipedia...")
    url = "https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia"
    tables = pd.read_html(url)
    
    # The first table usually contains the main list of tickers
    df = tables[0]
    
    # Validasi jika nama kolomnya mengandung 'Kode' atau sejenisnya
    ticker_col = None
    for col in df.columns:
        if 'kode' in str(col).lower() or 'simbol' in str(col).lower():
            ticker_col = col
            break
            
    if ticker_col is None:
        # Fallback to first column assuming it's the ticker
        ticker_col = df.columns[0]
        
    tickers = df[ticker_col].dropna().astype(str).tolist()
    
    # Filter 4-ltter valid tickers and append .JK
    valid_tickers = [f"{t.strip()}.JK" for t in tickers if len(t.strip()) == 4 and t.strip().isalpha()]
    print(f"Berhasil mengumpulkan {len(valid_tickers)} kode saham.")
    return valid_tickers

def filter_by_liquidity(tickers, min_turnover_idr=2_000_000_000, days=5):
    """
    Filter tickers based on average daily turnover (Volume * Price).
    min_turnover_idr: Minimum transaction value per day in Rupiah (Default: 2 Miliar)
    """
    print(f"Memfilter {len(tickers)} saham dengan likuiditas minimum Rp {min_turnover_idr/1e9:,.0f} Miliar/hari...")
    
    # Download batch of data for the last few days to check recent liquidity
    # Note: Downloading all 900 tickers history can take a few minutes.
    valid_universe = []
    
    # We do a quick download just for Close price and Volume
    data = yf.download(" ".join(tickers), period=f"{days}d", interval="1d", group_by='ticker', threads=True)
    
    for ticker in tickers:
        if ticker in data:
            df = data[ticker]
            if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
                continue
                
            # Drop NaN rows
            df = df.dropna(subset=['Close', 'Volume'])
            if len(df) == 0:
                continue
                
            # Calculate daily turnover (Price * Volume)
            # Karena harga saham dalam IDR dan volume dalam jumlah lembar, kalikan.
            # Rata-rata 5 hari terakhir
            df['Turnover'] = df['Close'] * df['Volume']
            avg_turnover = df['Turnover'].mean()
            
            if avg_turnover >= min_turnover_idr:
                valid_universe.append(ticker)
                
    print(f"Filter Selesai! Tersisa {len(valid_universe)} saham likuid yang layak untuk di-training.")
    return valid_universe

if __name__ == "__main__":
    all_tickers = get_all_idx_tickers()
    
    # For testing speed we can limit the test to the first 50 symbols if you want (uncomment below)
    # all_tickers = all_tickers[:50]
    
    liquid_tickers = filter_by_liquidity(all_tickers)
    
    # Save the liquid tickers to a file so we don't have to filter again next time.
    filepath = os.path.join(config.DATA_DIR, 'liquid_universe.txt')
    with open(filepath, 'w') as f:
        for t in liquid_tickers:
            f.write(f"{t}\n")
            
    print(f"Daftar tersimpan di {filepath}")
