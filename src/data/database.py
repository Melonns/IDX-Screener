"""
database.py — DatabaseManager untuk IDX-Screener v2.

Mengelola koneksi SQLite dan semua operasi CRUD untuk:
- daily_prices: raw OHLCV data
- daily_indicators: pre-computed technical indicators (feature store)
- signals: output scoring engine
- signal_outcomes: performance tracking per sinyal
- stocks: metadata saham
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


SCHEMA_SQL = """
-- Metadata saham
-- avg_volume_20d TIDAK disimpan di sini (stale tiap hari).
-- Ambil on-the-fly dari daily_indicators saat dibutuhkan.
CREATE TABLE IF NOT EXISTS stocks (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    last_updated TEXT
);

-- Data harga harian OHLCV (raw, tidak pernah dimodifikasi setelah insert)
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    volume          REAL,
    is_valid        INTEGER NOT NULL DEFAULT 1,
    validation_note TEXT,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_date   ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker);

-- Indikator teknikal terhitung per hari (derived, bisa dihitung ulang)
-- Ini adalah feature store untuk scoring engine dan ML training.
CREATE TABLE IF NOT EXISTS daily_indicators (
    ticker           TEXT NOT NULL,
    date             TEXT NOT NULL,
    rsi_14           REAL,
    ema_9            REAL,
    ema_21           REAL,
    ema_50           REAL,
    macd             REAL,
    macd_signal      REAL,
    macd_diff        REAL,
    bb_upper         REAL,
    bb_lower         REAL,
    bb_width         REAL,
    atr_14           REAL,
    volume_ratio_20d REAL,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker, date) REFERENCES daily_prices(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_indicators_date   ON daily_indicators(date);
CREATE INDEX IF NOT EXISTS idx_daily_indicators_ticker ON daily_indicators(ticker);

-- Contextual / Cross-Sectional Features (Phase 4)
CREATE TABLE IF NOT EXISTS contextual_indicators (
    ticker                 TEXT NOT NULL,
    date                   TEXT NOT NULL,
    rel_strength_5d        REAL,
    rel_strength_5d_rank   REAL,
    vol_accum_5d           REAL,
    vol_accum_5d_rank      REAL,
    turnover_5d            REAL,
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker, date) REFERENCES daily_prices(ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_contextual_date   ON contextual_indicators(date);
CREATE INDEX IF NOT EXISTS idx_contextual_ticker ON contextual_indicators(ticker);

-- Output scoring engine per saham per hari
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    skor_total      INTEGER NOT NULL,
    sinyal          TEXT    NOT NULL CHECK(sinyal IN ('BULLISH', 'BEARISH', 'NEUTRAL')),
    breakdown       TEXT,   -- JSON string
    scoring_version TEXT,   -- versi rule, penting buat compare antar versi
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_date   ON signals(date);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);

-- Tracking performa sinyal setelah N hari
-- target_threshold disimpan karena sering dituning.
-- PENTING: threshold minimum harus di atas total biaya transaksi
-- (~0.3-0.5% roundtrip untuk IDX). Kalau threshold < biaya, hasil bisa
-- terlihat profit padahal setelah fee tetap merugi.
CREATE TABLE IF NOT EXISTS signal_outcomes (
    signal_id        INTEGER NOT NULL REFERENCES signals(id),
    ticker           TEXT    NOT NULL,
    signal_date      TEXT    NOT NULL,
    price_at_signal  REAL,
    price_n1         REAL,
    return_n1        REAL,
    price_n3         REAL,
    return_n3        REAL,
    price_n5         REAL,
    return_n5        REAL,
    target_threshold REAL,
    hit_target       INTEGER,
    PRIMARY KEY (signal_id)
);
"""


class DatabaseManager:
    """
    Mengelola SQLite database untuk IDX-Screener v2.

    Semua operasi database dilakukan melalui class ini.
    Gunakan sebagai context manager atau inisialisasi langsung.

    Contoh:
        db = DatabaseManager('idx_screener.db')
        db.save_prices('BBCA.JK', df)
        prices = db.get_prices('BBCA.JK', '2025-01-01', '2026-01-01')
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        """Context manager untuk koneksi SQLite dengan WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Inisialisasi schema database. Aman dipanggil berulang kali."""
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ─────────────────────────────────────────────────────────────
    # STOCKS
    # ─────────────────────────────────────────────────────────────

    def upsert_stock(self, ticker: str, name: str = None, sector: str = None) -> None:
        """Insert atau update metadata saham."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stocks (ticker, name, sector, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name = excluded.name,
                    sector = excluded.sector,
                    last_updated = excluded.last_updated
                """,
                (ticker, name, sector, datetime.now().isoformat()),
            )

    # ─────────────────────────────────────────────────────────────
    # DAILY PRICES
    # ─────────────────────────────────────────────────────────────

    def save_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Simpan OHLCV data ke daily_prices.
        DataFrame harus punya kolom: Open, High, Low, Close, Volume.
        Index harus berupa DatetimeIndex atau string date.

        Baris yang sudah ada akan di-skip (INSERT OR IGNORE).
        Validasi data dilakukan di sini.

        Returns:
            Jumlah baris yang berhasil diinsert.
        """
        if df.empty:
            return 0

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame harus punya kolom: {required_cols}")

        rows = []
        for date_idx, row in df.iterrows():
            date_str = str(date_idx)[:10]  # YYYY-MM-DD
            is_valid, note = self._validate_candle(row)
            rows.append((
                ticker,
                date_str,
                float(row['Open'])   if pd.notna(row['Open'])   else None,
                float(row['High'])   if pd.notna(row['High'])   else None,
                float(row['Low'])    if pd.notna(row['Low'])    else None,
                float(row['Close'])  if pd.notna(row['Close'])  else None,
                float(row['Volume']) if pd.notna(row['Volume']) else None,
                1 if is_valid else 0,
                note,
            ))

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO daily_prices
                    (ticker, date, open, high, low, close, volume, is_valid, validation_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            # Update metadata
            conn.execute(
                "INSERT OR REPLACE INTO stocks (ticker, last_updated) VALUES (?, ?)",
                (ticker, datetime.now().isoformat()),
            )

        return len(rows)

    def _validate_candle(self, row) -> tuple[bool, Optional[str]]:
        """
        Validasi satu candle. Return (is_valid, reason).
        Row yang invalid tetap disimpan tapi is_valid=0.
        """
        open_  = row.get('Open')
        high   = row.get('High')
        low    = row.get('Low')
        close  = row.get('Close')
        volume = row.get('Volume')

        if any(pd.isna(v) for v in [open_, high, low, close]):
            return False, "Missing OHLC value"
        if high < low:
            return False, f"high ({high:.2f}) < low ({low:.2f})"
        if close <= 0 or open_ <= 0:
            return False, f"Non-positive price: close={close}, open={open_}"
        if pd.notna(volume) and volume == 0:
            return False, "Zero volume"

        return True, None

    def get_prices(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
        valid_only: bool = True,
    ) -> pd.DataFrame:
        """
        Ambil OHLCV dari daily_prices.

        Args:
            ticker: Kode saham (misal 'BBCA.JK')
            start: Tanggal awal 'YYYY-MM-DD' (opsional)
            end: Tanggal akhir 'YYYY-MM-DD' (opsional)
            valid_only: Kalau True, hanya return baris dengan is_valid=1

        Returns:
            DataFrame dengan index tanggal, kolom Open/High/Low/Close/Volume.
        """
        conditions = ["ticker = ?"]
        params: list = [ticker]

        if start:
            conditions.append("date >= ?")
            params.append(start)
        if end:
            conditions.append("date <= ?")
            params.append(end)
        if valid_only:
            conditions.append("is_valid = 1")

        where = " AND ".join(conditions)
        query = f"""
            SELECT date, open AS Open, high AS High, low AS Low,
                   close AS Close, volume AS Volume
            FROM daily_prices
            WHERE {where}
            ORDER BY date ASC
        """

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])

        if not df.empty:
            df = df.set_index('date')

        return df

    def get_latest_date(self, ticker: str) -> Optional[str]:
        """Return tanggal data terbaru untuk ticker, atau None kalau belum ada."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM daily_prices WHERE ticker = ?", (ticker,)
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_tickers(self) -> list[str]:
        """Return semua ticker yang ada di database."""
        with self._connect() as conn:
            rows = conn.execute("SELECT ticker FROM stocks ORDER BY ticker").fetchall()
        return [r[0] for r in rows]

    # ─────────────────────────────────────────────────────────────
    # DAILY INDICATORS
    # ─────────────────────────────────────────────────────────────

    def save_indicators(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Simpan pre-computed technical indicators ke daily_indicators.
        DataFrame harus punya kolom sesuai schema.
        INSERT OR REPLACE — akan overwrite kalau sudah ada.

        Returns:
            Jumlah baris yang disimpan.
        """
        if df.empty:
            return 0

        indicator_cols = [
            'rsi_14', 'ema_9', 'ema_21', 'ema_50',
            'macd', 'macd_signal', 'macd_diff',
            'bb_upper', 'bb_lower', 'bb_width',
            'atr_14', 'volume_ratio_20d',
        ]

        rows = []
        for date_idx, row in df.iterrows():
            date_str = str(date_idx)[:10]
            values = [
                float(row[col]) if col in row.index and pd.notna(row[col]) else None
                for col in indicator_cols
            ]
            rows.append((ticker, date_str, *values))

        placeholders = ", ".join(["?"] * (2 + len(indicator_cols)))
        cols_str = "ticker, date, " + ", ".join(indicator_cols)

        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO daily_indicators ({cols_str}) VALUES ({placeholders})",
                rows,
            )

        return len(rows)

    def get_indicators(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Ambil pre-computed indicators dari daily_indicators.
        Return DataFrame dengan index tanggal.
        """
        conditions = ["ticker = ?"]
        params: list = [ticker]

        if start:
            conditions.append("date >= ?")
            params.append(start)
        if end:
            conditions.append("date <= ?")
            params.append(end)

        where = " AND ".join(conditions)
        query = f"""
            SELECT date, rsi_14, ema_9, ema_21, ema_50,
                   macd, macd_signal, macd_diff,
                   bb_upper, bb_lower, bb_width,
                   atr_14, volume_ratio_20d
            FROM daily_indicators
            WHERE {where}
            ORDER BY date ASC
        """

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])

        if not df.empty:
            df = df.set_index('date')

        return df

    def get_prices_with_indicators(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Return JOIN dari daily_prices + daily_indicators.
        Berguna untuk scoring engine dan backtest.
        Hanya return baris dengan is_valid=1.
        """
        conditions = ["p.ticker = ?"]
        params: list = [ticker]

        if start:
            conditions.append("p.date >= ?")
            params.append(start)
        if end:
            conditions.append("p.date <= ?")
            params.append(end)

        conditions.append("p.is_valid = 1")
        where = " AND ".join(conditions)

        query = f"""
            SELECT
                p.date,
                p.open  AS Open,
                p.high  AS High,
                p.low   AS Low,
                p.close AS Close,
                p.volume AS Volume,
                i.rsi_14, i.ema_9, i.ema_21, i.ema_50,
                i.macd, i.macd_signal, i.macd_diff,
                i.bb_upper, i.bb_lower, i.bb_width,
                i.atr_14, i.volume_ratio_20d
            FROM daily_prices p
            LEFT JOIN daily_indicators i USING (ticker, date)
            WHERE {where}
            ORDER BY p.date ASC
        """

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])

        if not df.empty:
            df = df.set_index('date')

        return df

    # ─────────────────────────────────────────────────────────────
    # CONTEXTUAL INDICATORS (PHASE 4)
    # ─────────────────────────────────────────────────────────────

    def save_contextual_indicators(self, ticker: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        
        cols = ['rel_strength_5d', 'rel_strength_5d_rank', 'vol_accum_5d', 'vol_accum_5d_rank', 'turnover_5d']
        rows = []
        for date_idx, row in df.iterrows():
            date_str = str(date_idx)[:10]
            values = [float(row[c]) if c in row.index and pd.notna(row[c]) else None for c in cols]
            rows.append((ticker, date_str, *values))
            
        placeholders = ", ".join(["?"] * (2 + len(cols)))
        cols_str = "ticker, date, " + ", ".join(cols)
        
        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO contextual_indicators ({cols_str}) VALUES ({placeholders})",
                rows
            )
        return len(rows)

    def get_prices_with_context(self, ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
        """JOIN daily_prices, market_index, and contextual_indicators."""
        conditions = ["p.ticker = ?"]
        params: list = [ticker]
        if start:
            conditions.append("p.date >= ?")
            params.append(start)
        if end:
            conditions.append("p.date <= ?")
            params.append(end)
        conditions.append("p.is_valid = 1")
        where = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                p.date, p.open AS Open, p.high AS High, p.low AS Low, p.close AS Close, p.volume AS Volume,
                c.rel_strength_5d, c.rel_strength_5d_rank, c.vol_accum_5d, c.vol_accum_5d_rank, c.turnover_5d,
                m.ret_5d AS ihsg_ret_5d, m.slope_20d AS ihsg_slope_20d
            FROM daily_prices p
            LEFT JOIN contextual_indicators c USING (ticker, date)
            LEFT JOIN market_index m ON p.date = m.date
            WHERE {where}
            ORDER BY p.date ASC
        """
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
        if not df.empty:
            df = df.set_index('date')
        return df

    # ─────────────────────────────────────────────────────────────
    # SIGNALS
    # ─────────────────────────────────────────────────────────────

    def save_signal(self, result: dict) -> int:
        """
        Simpan hasil scoring ke tabel signals.

        Args:
            result: Dict output dari ScoringEngine.score(), dengan kunci:
                    kode, tanggal, skor_total, sinyal, breakdown, scoring_version

        Returns:
            ID sinyal yang baru disimpan (untuk link ke signal_outcomes).
        """
        breakdown_json = json.dumps(result.get('breakdown', []), ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals (ticker, date, skor_total, sinyal, breakdown, scoring_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result['kode'],
                    result['tanggal'],
                    result['skor_total'],
                    result['sinyal'],
                    breakdown_json,
                    result.get('scoring_version', 'rule_v1.0'),
                    datetime.now().isoformat(),
                ),
            )
        return cursor.lastrowid

    def get_signals(
        self,
        ticker: str = None,
        date: str = None,
        min_score: int = None,
    ) -> pd.DataFrame:
        """Ambil sinyal dengan filter opsional. Return DataFrame."""
        conditions = []
        params: list = []

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if date:
            conditions.append("date = ?")
            params.append(date)
        if min_score is not None:
            conditions.append("skor_total >= ?")
            params.append(min_score)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT id, ticker, date, skor_total, sinyal, breakdown, scoring_version, created_at
            FROM signals
            {where}
            ORDER BY date DESC, skor_total DESC
        """

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_pending_outcomes(self, days_ago_min: int = 1, days_ago_max: int = 7) -> pd.DataFrame:
        """
        Return sinyal yang belum punya outcome dan sudah cukup lama
        (antara days_ago_min dan days_ago_max hari lalu).
        """
        query = """
            SELECT s.id, s.ticker, s.date, s.skor_total
            FROM signals s
            LEFT JOIN signal_outcomes o ON s.id = o.signal_id
            WHERE o.signal_id IS NULL
              AND date(s.date) <= date('now', ? || ' days')
              AND date(s.date) >= date('now', ? || ' days')
            ORDER BY s.date ASC
        """
        params = [f'-{days_ago_min}', f'-{days_ago_max}']
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    # ─────────────────────────────────────────────────────────────
    # SIGNAL OUTCOMES
    # ─────────────────────────────────────────────────────────────

    def save_outcome(self, outcome: dict) -> None:
        """
        Simpan atau update outcome sebuah sinyal.

        Args:
            outcome: Dict dengan kunci: signal_id, ticker, signal_date,
                     price_at_signal, price_n1, return_n1, price_n3,
                     return_n3, price_n5, return_n5, target_threshold, hit_target
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO signal_outcomes
                    (signal_id, ticker, signal_date, price_at_signal,
                     price_n1, return_n1, price_n3, return_n3,
                     price_n5, return_n5, target_threshold, hit_target)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome['signal_id'],
                    outcome['ticker'],
                    outcome['signal_date'],
                    outcome.get('price_at_signal'),
                    outcome.get('price_n1'),
                    outcome.get('return_n1'),
                    outcome.get('price_n3'),
                    outcome.get('return_n3'),
                    outcome.get('price_n5'),
                    outcome.get('return_n5'),
                    outcome.get('target_threshold'),
                    outcome.get('hit_target'),
                ),
            )

    def get_outcomes(self, min_score: int = None) -> pd.DataFrame:
        """Ambil semua outcome yang sudah ter-fill, join dengan signals."""
        where = ""
        params: list = []
        if min_score is not None:
            where = "WHERE s.skor_total >= ?"
            params.append(min_score)

        query = f"""
            SELECT
                s.ticker, s.date AS signal_date, s.skor_total, s.sinyal,
                o.price_at_signal, o.return_n1, o.return_n3, o.return_n5,
                o.target_threshold, o.hit_target
            FROM signal_outcomes o
            JOIN signals s ON o.signal_id = s.id
            {where}
            ORDER BY s.date DESC
        """
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    # ─────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────

    def get_db_stats(self) -> dict:
        """Return statistik isi database untuk debugging."""
        with self._connect() as conn:
            stats = {}
            for table in ['stocks', 'daily_prices', 'daily_indicators', 'signals', 'signal_outcomes']:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[table] = row[0]
            invalid = conn.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE is_valid = 0"
            ).fetchone()[0]
            stats['invalid_prices'] = invalid
        return stats
