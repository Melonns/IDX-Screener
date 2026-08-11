"""
forward_tracker.py — Production Forward-Tracking & Paper Trading Engine (Phase 7)

Strategi Terkunci: Dividend Cum-Date Drift v1.0
  - Entry Window : T-10 trading day sebelum Cum-Date
  - Exit Point   : Cum-Date Close (T-1 sebelum Ex-Date)
  - Parameter    : Yield >= 4.0%, Turnover 5D >= 1 Miliar IDR
  - Fee Model    : 0.40% Roundtrip
  - Scoring Ver  : V3_DIVIDEND_DRIFT_LOCKED

Fungsi Utama:
1. `scan_upcoming_signals()` : Mendeteksi saham yang HARI INI berada di T-10 window dividend qualified.
2. `record_paper_signals()` : Menyimpan sinyal aktif ke tabel `signals` & `signal_outcomes`.
3. `update_active_outcomes()`: Memperbarui return real-time dari sinyal aktif saat hari berjalan.
"""

import sys, os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
import config as app_config

SCORING_VERSION = "V3_DIVIDEND_DRIFT_LOCKED"
MIN_YIELD = 4.0
MIN_TURNOVER = 1_000_000_000
WINDOW_DAYS = 10
FEE = 0.004

class DividendForwardTracker:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def scan_upcoming_signals(self, as_of_date: str = None) -> pd.DataFrame:
        """
        Scan stock universe for active dividend drift signals on a given date.
        If as_of_date is None, uses today's date.
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y-%m-%d')

        as_of_dt = pd.to_datetime(as_of_date)

        with self.db._connect() as conn:
            # 1. Fetch upcoming / recent dividends with yield >= 4.0%
            divs = pd.read_sql_query(f"""
                SELECT ticker, date AS ex_date, value AS div_amount, dividend_yield
                FROM corporate_actions
                WHERE event_type = 'DIVIDEND'
                AND dividend_yield >= {MIN_YIELD}
                AND date >= '{as_of_date}'
                ORDER BY date ASC
            """, conn)

            # 2. Fetch prices unadjusted
            prices = pd.read_sql_query("""
                SELECT ticker, date, close, volume
                FROM prices_unadj
                ORDER BY ticker, date ASC
            """, conn)

        if divs.empty:
            print(f"[{as_of_date}] Tidak ada event dividen qualified (Yield >= {MIN_YIELD}%) mendatang.")
            return pd.DataFrame()

        prices['date'] = pd.to_datetime(prices['date'])
        prices = prices.sort_values(['ticker', 'date'])
        prices['turnover'] = prices['close'] * prices['volume']
        prices['turnover_5d'] = prices.groupby('ticker')['turnover'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )

        active_signals = []

        for _, row in divs.iterrows():
            ticker  = row['ticker']
            ex_date = pd.to_datetime(row['ex_date'])
            yield_v = row['dividend_yield']

            p_sub = prices[prices['ticker'] == ticker].sort_values('date')
            before_ex = p_sub[p_sub['date'] < ex_date]

            if len(before_ex) < WINDOW_DAYS + 1:
                continue

            cum_date_row = before_ex.iloc[-1]
            entry_row    = before_ex.iloc[-(WINDOW_DAYS + 1)]

            entry_dt = entry_row['date']
            cum_dt   = cum_date_row['date']

            # Liquidity check
            t_5d = entry_row['turnover_5d']
            if pd.isna(t_5d) or t_5d < MIN_TURNOVER:
                continue

            # Check if as_of_date falls exactly on entry_dt or within active window
            is_entry_day = (as_of_dt.strftime('%Y-%m-%d') == entry_dt.strftime('%Y-%m-%d'))
            is_active    = (entry_dt <= as_of_dt <= cum_dt)

            active_signals.append({
                'as_of_date':  as_of_date,
                'ticker':      ticker,
                'entry_date':  entry_dt.strftime('%Y-%m-%d'),
                'cum_date':    cum_dt.strftime('%Y-%m-%d'),
                'ex_date':     ex_date.strftime('%Y-%m-%d'),
                'yield':       yield_v,
                'entry_price': entry_row['close'],
                'turnover_5d': t_5d,
                'is_entry_today': is_entry_day,
                'is_active':   is_active
            })

        df_res = pd.DataFrame(active_signals)
        return df_res

    def record_and_update_signals(self, as_of_date: str = None) -> int:
        """
        Record new signals into SQLite database `signals` & `signal_outcomes`.
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y-%m-%d')

        df_signals = self.scan_upcoming_signals(as_of_date)
        if df_signals.empty:
            return 0

        saved = 0
        with self.db._connect() as conn:
            for _, r in df_signals.iterrows():
                if not r['is_active']:
                    continue

                # Check if signal already recorded
                existing = conn.execute("""
                    SELECT id FROM signals
                    WHERE ticker = ? AND date = ? AND scoring_version = ?
                """, (r['ticker'], r['entry_date'], SCORING_VERSION)).fetchone()

                if not existing:
                    breakdown = {
                        'yield': r['yield'],
                        'cum_date': r['cum_date'],
                        'ex_date': r['ex_date'],
                        'turnover_5d': r['turnover_5d']
                    }
                    cursor = conn.execute("""
                        INSERT INTO signals (ticker, date, skor_total, sinyal, breakdown, scoring_version, created_at)
                        VALUES (?, ?, 100, 'BULLISH', ?, ?, ?)
                    """, (
                        r['ticker'],
                        r['entry_date'],
                        pd.io.json.dumps(breakdown),
                        SCORING_VERSION,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    sig_id = cursor.lastrowid

                    conn.execute("""
                        INSERT INTO signal_outcomes (signal_id, ticker, signal_date, price_at_signal, target_threshold)
                        VALUES (?, ?, ?, ?, ?)
                    """, (sig_id, r['ticker'], r['entry_date'], r['entry_price'], 0.003))
                    saved += 1

            conn.commit()

        print(f"[{as_of_date}] Recorded {saved} new paper trading signals into SQLite DB.")
        return saved


if __name__ == '__main__':
    db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    tracker = DividendForwardTracker(db)

    # Demo run as of today
    print("="*75)
    print("  IDX-SCREENER V3 — FORWARD TRACKING & PAPER TRADING ENGINE")
    print("="*75)
    df_act = tracker.scan_upcoming_signals()
    if not df_act.empty:
        print("\nSinyal Aktif & Mendatang:")
        print(df_act[['ticker', 'entry_date', 'cum_date', 'yield', 'turnover_5d', 'is_active']].to_string(index=False))
    else:
        print("\nTidak ada sinyal aktif hari ini.")
