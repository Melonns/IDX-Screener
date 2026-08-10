"""
extract_yfinance_dividends.py — Trik Ekstraksi Data Dividen Historis via yfinance

Tujuan: Mengambil data historis tanggal & nominal dividen untuk seluruh 45 saham di database
menggunakan yfinance (100% gratis, stabil, tanpa terhalang 503/403).
"""
import sys, os
sys.path.insert(0, 'src')
import pandas as pd
import yfinance as yf
from data.database import DatabaseManager
import config as app_config

db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))
tickers = db.get_tickers()

print("="*75)
print("  EKSTRAKSI DATA DIVIDEN HISTORIS VIA YFINANCE")
print(f"  Total Tickers : {len(tickers)}")
print("="*75)

div_records = []

for ticker_code in tickers:
    try:
        t = yf.Ticker(ticker_code)
        divs = t.dividends
        if not divs.empty:
            div_df = divs.reset_index()
            # Handle if divs is DataFrame or Series
            val_col = 'Dividends' if 'Dividends' in div_df.columns else div_df.columns[1]
            
            for idx, row in div_df.iterrows():
                date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                val = float(row[val_col])
                div_records.append({
                    'ticker': ticker_code,
                    'date': date_str,
                    'dividend_per_share': val
                })
            last_row = div_df.iloc[-1]
            last_date = pd.to_datetime(last_row['date']).strftime('%Y-%m-%d')
            last_val = float(last_row[val_col])
            print(f"  [✓] {ticker_code:<10} | {len(div_df):2d} dividen terdeteksi | Dividen terakhir: {last_date} (Rp {last_val:,.1f})")
        else:
            print(f"  [-] {ticker_code:<10} | Tidak ada record dividen")
    except Exception as e:
        print(f"  [!] {ticker_code:<10} | Error: {e}")

df_divs = pd.DataFrame(div_records)
print(f"\nTotal Record Dividen Terkumpul: {len(df_divs):,} baris data.")

if not df_divs.empty:
    print("\nSample Data Dividen (10 Terakhir):")
    print(df_divs.tail(10).to_string(index=False))
