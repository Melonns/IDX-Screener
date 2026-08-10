"""
provider.py — DataProvider abstraction layer untuk IDX-Screener v2.

Pattern: interface DataProvider → implementasi YFinanceProvider.
Desain ini memungkinkan ganti data source (Stockbit, RTI, dll) nanti
tanpa ubah satu baris pun di scoring engine atau backtest.

Flow:
    provider.get_or_fetch(ticker) → cek SQLite dulu → fetch yfinance
                                    kalau data belum ada atau ketinggalan
"""

import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .database import DatabaseManager


class DataProvider(ABC):
    """
    Abstract interface untuk semua data source.

    Setiap implementasi harus menyediakan:
    - fetch(ticker, start, end) -> pd.DataFrame   raw download
    - get_or_fetch(ticker, ...)  -> pd.DataFrame  cek cache dulu
    """

    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Download raw OHLCV data dari source.
        Return DataFrame dengan kolom Open/High/Low/Close/Volume dan DatetimeIndex.
        """
        ...

    @abstractmethod
    def get_or_fetch(
        self,
        ticker: str,
        period_days: int = 365,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV data. Cek database cache dulu,
        hanya fetch dari source kalau data belum ada atau sudah ketinggalan.
        """
        ...


class YFinanceProvider(DataProvider):
    """
    DataProvider berbasis yfinance + SQLite cache.

    Strategi caching:
    - Cek tanggal data terbaru di SQLite.
    - Kalau data sudah ada dan up-to-date (gap ≤ max_gap_days hari bursa),
      return dari SQLite.
    - Kalau ada gap, fetch hanya data yang belum ada dari yfinance.
    - Data yang baru di-fetch divalidasi dan disimpan ke SQLite.

    Args:
        db: DatabaseManager instance yang sudah terhubung ke SQLite.
        max_gap_days: Toleransi gap sebelum trigger re-fetch (default 1 hari bursa).
        request_delay: Jeda antar request yfinance untuk hindari rate limiting.
        max_retries: Jumlah retry jika fetch gagal.
    """

    def __init__(
        self,
        db: DatabaseManager,
        max_gap_days: int = 1,
        request_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        self.db = db
        self.max_gap_days = max_gap_days
        self.request_delay = request_delay
        self.max_retries = max_retries

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Download OHLCV dari yfinance dengan retry logic.

        Args:
            ticker: Kode saham (misal 'BBCA.JK')
            start: Tanggal awal 'YYYY-MM-DD'
            end: Tanggal akhir 'YYYY-MM-DD'

        Returns:
            DataFrame dengan kolom Open/High/Low/Close/Volume.
            Empty DataFrame jika data tidak ditemukan.
        """
        formats_to_try = self._get_ticker_formats(ticker)
        last_error: Optional[str] = None

        for attempt in range(self.max_retries):
            if attempt > 0:
                time.sleep(self.request_delay * (1 + attempt))

            for fmt in formats_to_try:
                try:
                    df = yf.download(
                        fmt,
                        start=start,
                        end=end,
                        interval='1d',
                        progress=False,
                        auto_adjust=True,
                        timeout=15,
                    )

                    if df.empty:
                        last_error = f"Empty DataFrame for {fmt}"
                        continue

                    # Handle MultiIndex columns (yfinance quirk untuk single ticker)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)

                    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
                    if not required.issubset(df.columns):
                        last_error = f"Kolom tidak lengkap untuk {fmt}: {df.columns.tolist()}"
                        continue

                    return df[list(required)].copy()

                except Exception as exc:
                    last_error = str(exc)
                    continue

        print(f"[YFinanceProvider] Warning: Gagal fetch {ticker} setelah {self.max_retries} attempts. "
              f"Last error: {last_error}")
        return pd.DataFrame()

    def get_or_fetch(
        self,
        ticker: str,
        period_days: int = 365 * 3,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV data untuk ticker, dengan cache-first strategy.

        Langkah:
        1. Cek tanggal data terbaru di SQLite.
        2. Kalau force_refresh=True atau data belum ada, fetch penuh.
        3. Kalau data ada tapi ada gap, fetch hanya rentang yang hilang.
        4. Simpan data baru ke SQLite (validasi otomatis di DatabaseManager).
        5. Return data lengkap dari SQLite (valid only).

        Args:
            ticker: Kode saham
            period_days: Berapa hari ke belakang yang diinginkan (default 3 tahun)
            force_refresh: Paksa re-fetch meski data sudah ada

        Returns:
            DataFrame OHLCV dengan DatetimeIndex.
        """
        today = date.today().isoformat()
        start_needed = (date.today() - timedelta(days=period_days)).isoformat()

        latest_in_db = self.db.get_latest_date(ticker)
        needs_fetch = (
            force_refresh
            or latest_in_db is None
            or self._has_gap(latest_in_db, today)
        )

        if needs_fetch:
            # Fetch hanya dari hari setelah data terakhir (incremental)
            fetch_start = start_needed if latest_in_db is None else latest_in_db
            print(f"[YFinanceProvider] Fetching {ticker} dari {fetch_start} hingga {today}...")

            time.sleep(self.request_delay)
            raw_df = self.fetch(ticker, fetch_start, today)

            if not raw_df.empty:
                saved = self.db.save_prices(ticker, raw_df)
                print(f"[YFinanceProvider] {ticker}: {saved} baris disimpan ke database.")
            else:
                print(f"[YFinanceProvider] Warning: Tidak ada data baru untuk {ticker}.")
        else:
            print(f"[YFinanceProvider] {ticker}: Data sudah up-to-date (latest: {latest_in_db}).")

        return self.db.get_prices(ticker, start=start_needed, valid_only=True)

    def get_or_fetch_batch(
        self,
        tickers: list[str],
        period_days: int = 365 * 3,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch beberapa ticker sekaligus.
        Jeda antar request untuk hindari rate limiting yfinance.

        Returns:
            Dict {ticker: DataFrame}. Ticker yang gagal punya empty DataFrame.
        """
        results = {}
        total = len(tickers)

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[Batch {i}/{total}] Memproses {ticker}...")
            try:
                df = self.get_or_fetch(ticker, period_days=period_days, force_refresh=force_refresh)
                results[ticker] = df
            except Exception as exc:
                print(f"[YFinanceProvider] Error untuk {ticker}: {exc}")
                results[ticker] = pd.DataFrame()

        return results

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _get_ticker_formats(self, ticker: str) -> list[str]:
        """
        Return daftar format ticker yang akan dicoba (berurutan).
        Contoh: 'BBCA' → ['BBCA.JK', 'BBCA']
                'BBCA.JK' → ['BBCA.JK', 'BBCA']
        """
        ticker = ticker.upper().strip()
        if ticker.endswith('.JK'):
            return [ticker, ticker[:-3]]
        elif len(ticker) == 4 and ticker.isalpha():
            return [f'{ticker}.JK', ticker]
        return [ticker]

    def _has_gap(self, latest_date_str: str, today_str: str) -> bool:
        """
        Return True kalau gap antara data terbaru dan hari ini
        lebih dari max_gap_days hari kalender.
        Ini bukan hari bursa murni (sederhana tapi cukup untuk trigger re-fetch).
        """
        try:
            latest = datetime.strptime(latest_date_str[:10], '%Y-%m-%d').date()
            today = datetime.strptime(today_str[:10], '%Y-%m-%d').date()
            gap = (today - latest).days
            return gap > self.max_gap_days
        except (ValueError, TypeError):
            return True  # Kalau parse gagal, anggap perlu re-fetch
