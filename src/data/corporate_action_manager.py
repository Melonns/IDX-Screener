"""
corporate_action_manager.py — Manager & Ingestion untuk Aksi Korporasi (Phase 5)

Menarik data Dividen & Stock Split historis via yfinance, menghitung Dividend Yield,
dan menyimpannya ke tabel corporate_actions di SQLite.
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
import config as app_config


class CorporateActionManager:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def sync_dividends(self) -> int:
        tickers = self.db.get_tickers()
        print(f"Syncing Corporate Actions (Dividends & Splits) untuk {len(tickers)} tickers...")

        rows_to_save = []

        for ticker in tqdm(tickers, desc="Fetching YFinance Actions"):
            try:
                t = yf.Ticker(ticker)
                divs = t.dividends
                
                if divs.empty:
                    continue

                div_df = divs.reset_index()
                val_col = 'Dividends' if 'Dividends' in div_df.columns else div_df.columns[1]

                # Load prices for this ticker to compute yield
                prices_df = self.db.get_prices(ticker, valid_only=True)
                if prices_df.empty:
                    continue
                
                prices_df.index = pd.to_datetime(prices_df.index).tz_localize(None)

                for idx, row in div_df.iterrows():
                    dt = pd.to_datetime(row['date']).tz_localize(None)
                    date_str = dt.strftime('%Y-%m-%d')
                    val = float(row[val_col])

                    # Find close price on or near dividend date
                    div_yield = None
                    if dt in prices_df.index:
                        close_p = float(prices_df.loc[dt, 'Close'])
                        if close_p > 0:
                            div_yield = (val / close_p) * 100.0
                    else:
                        sub_p = prices_df[prices_df.index <= dt]
                        if not sub_p.empty:
                            close_p = float(sub_p['Close'].iloc[-1])
                            if close_p > 0:
                                div_yield = (val / close_p) * 100.0

                    rows_to_save.append((ticker, date_str, 'DIVIDEND', val, div_yield))

            except Exception as exc:
                print(f"Error fetching {ticker}: {exc}")

        saved_count = self.db.save_corporate_actions(rows_to_save)
        print(f"Berhasil menyimpan {saved_count} record aksi korporasi ke database.")
        return saved_count


if __name__ == '__main__':
    db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    mgr = CorporateActionManager(db)
    mgr.sync_dividends()
