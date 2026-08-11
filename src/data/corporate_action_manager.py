"""
corporate_action_manager.py — REBUILT v2: Manager Aksi Korporasi dengan Yield Historis yang Benar

PERBAIKAN BUG KRITIS (v2):
- v1 menggunakan harga dari database SQLite lokal (hanya ada dari Agustus 2023)
  sehingga semua event dividen sebelum 2023 tersimpan dengan dividend_yield = NULL.
- v2 menggunakan yfinance t.history(start, end) untuk mengambil harga historis
  pada tanggal ex-date yang tepat, terlepas dari apakah ada di database lokal atau tidak.

CATATAN SURVIVORSHIP BIAS [WAJIB BACA]:
  Script ini menggunakan 45 saham yang SAAT INI berada di LQ45/database.
  Saham-saham ini adalah "pemenang" yang berhasil bertahan dan likuid hingga 2026.
  Komposisi LQ45 tahun 2015-2020 berbeda dari komposisi saat ini.
  Backtest menggunakan data yang diperpanjang ke belakang HARUS melaporkan
  keterbatasan survivorship bias ini secara eksplisit di setiap laporan.

RULES TIDAK BERUBAH:
  - Parameter entry/exit/yield threshold/turnover dikunci terpisah di pre-analysis protocol.
  - Script ini HANYA mengurus data collection, tidak menentukan parameter strategi.
"""
import sys, os
import time
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

    def _fetch_price_at_date(self, ticker: str, target_date: pd.Timestamp) -> float | None:
        """
        Fetch closing price at or just before target_date using yfinance history.
        Uses a 5-day window around target_date to handle weekends/holidays.
        Returns None if no data available.
        """
        start = (target_date - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        end   = (target_date + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                return None
            hist.index = hist.index.tz_localize(None) if hist.index.tzinfo else hist.index
            # Get last available close on or before target_date
            relevant = hist[hist.index <= target_date]
            if relevant.empty:
                return None
            return float(relevant['Close'].iloc[-1])
        except Exception:
            return None

    def sync_dividends_with_historical_prices(self, start_year: int = 2015) -> int:
        """
        Re-download dividend history and compute yield using HISTORICAL price
        at the time of each dividend event (ex-date), not current price.

        Args:
            start_year: Only include dividend events from this year onwards.
        """
        tickers = self.db.get_tickers()
        print(f"[CorporateActionManager v2] Syncing dividends dengan harga historis yang benar...")
        print(f"  Universe      : {len(tickers)} saham (LQ45 saat ini)")
        print(f"  Dari tahun    : {start_year}")
        print(f"  CATATAN       : Survivorship bias berlaku — lihat docstring file ini.")
        print()

        rows_to_save = []
        missing_price_count = 0

        for ticker in tqdm(tickers, desc="Fetching Historical Dividends"):
            try:
                t = yf.Ticker(ticker)
                divs = t.dividends

                if divs is None or (hasattr(divs, 'empty') and divs.empty):
                    continue

                div_df = divs.reset_index()
                div_df.columns = ['date', 'amount']
                div_df['date'] = pd.to_datetime(div_df['date']).dt.tz_localize(None)

                # Filter to start_year onwards
                div_df = div_df[div_df['date'].dt.year >= start_year]

                for _, row in div_df.iterrows():
                    ex_date = row['date']
                    div_amount = float(row['amount'])

                    if div_amount <= 0:
                        continue

                    # Fetch historical price at ex-date from yfinance (NOT from local DB)
                    close_price = self._fetch_price_at_date(ticker, ex_date)

                    if close_price is None or close_price <= 0:
                        missing_price_count += 1
                        div_yield = None
                    else:
                        div_yield = (div_amount / close_price) * 100.0

                    date_str = ex_date.strftime('%Y-%m-%d')
                    rows_to_save.append((ticker, date_str, 'DIVIDEND', div_amount, div_yield))

                # Rate limit: avoid hammering yfinance
                time.sleep(0.15)

            except Exception as exc:
                print(f"  Error fetching {ticker}: {exc}")

        print(f"\n  Total event dikumpulkan  : {len(rows_to_save)}")
        print(f"  Event tanpa harga historis: {missing_price_count} (tersimpan dengan yield=NULL)")

        saved_count = self.db.save_corporate_actions(rows_to_save)
        print(f"  Berhasil disimpan ke DB  : {saved_count} records")
        return saved_count


if __name__ == '__main__':
    db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    mgr = CorporateActionManager(db)

    # Rebuild dengan harga historis, dari 2015 ke atas.
    # Catatan: 2015 dipilih sebagai trade-off antara N yang memadai
    # dan keterbatasan survivorship bias yang masih bisa didokumentasikan.
    mgr.sync_dividends_with_historical_prices(start_year=2015)
