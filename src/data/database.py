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

-- Corporate Actions (Phase 5)
CREATE TABLE IF NOT EXISTS corporate_actions (
    ticker               TEXT NOT NULL,
    date                 TEXT NOT NULL,
    event_type           TEXT NOT NULL, -- DIVIDEND, SPLIT, BUYBACK, RIGHTS_ISSUE
    value                REAL,          -- Nominal dividen atau rasio split
    dividend_yield       REAL,          -- Dividend yield (% dari harga close saat itu)
    PRIMARY KEY (ticker, date, event_type)
);
CREATE INDEX IF NOT EXISTS idx_corp_action_date   ON corporate_actions(date);
CREATE INDEX IF NOT EXISTS idx_corp_action_ticker ON corporate_actions(ticker);

-- Pengumuman Keterbukaan Informasi IDX (Phase 5)
CREATE TABLE IF NOT EXISTS idx_announcements (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    announcement_id      TEXT UNIQUE,
    date                 TEXT NOT NULL,
    ticker               TEXT,
    title                TEXT NOT NULL,
    tags                 TEXT,
    summary              TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_announcement_date   ON idx_announcements(date);
CREATE INDEX IF NOT EXISTS idx_announcement_ticker ON idx_announcements(ticker);

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

-- Scan Results Persistence (untuk audit trail & histori)
CREATE TABLE IF NOT EXISTS scan_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    close           REAL,
    turnover_5d     REAL,
    unusual_count   INTEGER,
    unusual_tags    TEXT,   -- JSON string: list of unusual tag_ids
    all_tags        TEXT,   -- JSON string: all observation tags
    liquidity_note  TEXT,
    breadth_context TEXT,
    sector_context  TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_scan_results_date ON scan_results(scan_date);
CREATE INDEX IF NOT EXISTS idx_scan_results_ticker ON scan_results(ticker);
"""


class DatabaseManager:
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

    def save_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """Simpan OHLCV data ke daily_prices."""
        if df.empty:
            return 0

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame harus punya kolom: {required_cols}")

        rows = []
        for date_idx, row in df.iterrows():
            date_str = str(date_idx)[:10]
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
            conn.execute(
                "INSERT OR REPLACE INTO stocks (ticker, last_updated) VALUES (?, ?)",
                (ticker, datetime.now().isoformat()),
            )

        return len(rows)

    def _validate_candle(self, row) -> tuple[bool, Optional[str]]:
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
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM daily_prices WHERE ticker = ?", (ticker,)
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_tickers(self) -> list[str]:
        """Return semua ticker unik yang ada di daily_prices maupun stocks table."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM daily_prices UNION SELECT ticker FROM stocks ORDER BY ticker"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def save_indicators(self, ticker: str, df: pd.DataFrame) -> int:
        """Simpan pre-computed indicators ke daily_indicators dan update stocks table."""
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
            conn.execute(
                "INSERT OR REPLACE INTO stocks (ticker, last_updated) VALUES (?, ?)",
                (ticker, datetime.now().isoformat()),
            )

        return len(rows)

    def get_indicators(
        self,
        ticker: str,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
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
            SELECT p.date, p.open AS Open, p.high AS High, p.low AS Low,
                   p.close AS Close, p.volume AS Volume,
                   i.rsi_14, i.ema_9, i.ema_21, i.ema_50,
                   i.macd, i.macd_signal, i.macd_diff,
                   i.bb_upper, i.bb_lower, i.bb_width,
                   i.atr_14, i.volume_ratio_20d
            FROM daily_prices p
            LEFT JOIN daily_indicators i ON p.ticker = i.ticker AND p.date = i.date
            WHERE {where}
            ORDER BY p.date ASC
        """

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])

        if not df.empty:
            df = df.set_index('date')

        return df

    def get_db_stats(self) -> dict[str, int]:
        """Return row counts for all major tables."""
        tables = [
            'stocks', 'daily_prices', 'daily_indicators',
            'contextual_indicators', 'corporate_actions',
            'idx_announcements', 'signals', 'signal_outcomes',
            'scan_results',
        ]
        stats = {}
        with self._connect() as conn:
            for table in tables:
                try:
                    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    stats[table] = row[0] if row else 0
                except Exception:
                    stats[table] = 0
        return stats

    def save_corporate_actions(self, rows: list[tuple]) -> int:
        """Bulk save corporate action events (ticker, date, event_type, value, dividend_yield)."""
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO corporate_actions
                    (ticker, date, event_type, value, dividend_yield)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def save_scan_results(self, scan_date: str, results: list[dict]) -> int:
        """Persist scan results to scan_results table for audit trail."""
        if not results:
            return 0
        import json
        rows = []
        for item in results:
            unusual_tag_ids = [t['tag_id'] for t in item.get('unusual_tags', [])]
            all_tags_clean = []
            for t in item.get('all_tags', []):
                all_tags_clean.append({
                    'tag_id': t.get('tag_id', ''),
                    'description': t.get('description', ''),
                    'is_unusual': t.get('is_unusual', False),
                })
            rows.append((
                scan_date,
                item['ticker'],
                item.get('close'),
                item.get('turnover_5d'),
                item.get('unusual_count', 0),
                json.dumps(unusual_tag_ids),
                json.dumps(all_tags_clean),
                item.get('liquidity_note', ''),
                item.get('breadth_context', ''),
                item.get('sector_context', ''),
            ))
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO scan_results
                    (scan_date, ticker, close, turnover_5d, unusual_count,
                     unusual_tags, all_tags, liquidity_note, breadth_context, sector_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_recent_scan_results(self, scan_date: str = None, limit: int = 50) -> list[dict]:
        """Retrieve recent scan results, optionally filtered by date."""
        import json
        conditions = []
        params = []
        if scan_date:
            conditions.append("scan_date = ?")
            params.append(scan_date)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"""
            SELECT * FROM scan_results{where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d['unusual_tags'] = json.loads(d.get('unusual_tags') or '[]')
            d['all_tags'] = json.loads(d.get('all_tags') or '[]')
            results.append(d)
        return results
