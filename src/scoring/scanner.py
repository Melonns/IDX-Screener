"""
scanner.py — Daily Technical Observation Scanner (Descriptive, Non-Predictive)

Prinsip Utama:
- Deskriptif, bukan prediktif. Tidak ada skor total 0-100, tidak ada sinyal BULLISH/BEARISH/BUY/SELL.
- Menyaring saham berdasarkan Unusual Activity (aktivitas di luar kebiasaan relatif terhadap histori 60 hari saham itu sendiri).
- Cache-First & Capped Batch Fetch (Mencegah Gunicorn Timeout & menjaga scan selesai <15 detik).
- Thread-safe Progress Tracking untuk fitur /status & /progress Telegram.
- Parallel batch evaluation untuk performa scan yang lebih cepat (ThreadPoolExecutor).
- Terpisah secara eksplisit dari strategi yang sudah divalidasi (seperti Dividend Drift).
"""

import os, sys, time, threading
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
from data.provider import YFinanceProvider
from data.ingestion import compute_indicators
from data.idx_universe import fetch_live_idx_tickers, ALL_IDX_800_TICKERS
from scoring.observation_tags import evaluate_observation_tags, get_liquidity_note
from scoring.rarity_context import get_rarity_context, get_condition_streak
from scoring.market_breadth import get_market_breadth_context, get_sector_context
import config as app_config

MANDATORY_DISCLAIMER = """
⚠️  Ringkasan aktivitas teknikal harian, BUKAN sinyal beli/jual.
Riset internal membuktikan indikator teknikal murni tidak memprediksi arah harga ke depan.
Gunakan sebagai titik awal riset manual (berita, laporan keuangan, kondisi sektor).
""".strip()

# Max workers for parallel ticker evaluation
MAX_EVAL_WORKERS = 8


class TechnicalObservationScanner:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.provider = YFinanceProvider(db)

        # Thread-safe Progress Tracking
        self.progress_lock = threading.Lock()
        self.progress_data = {
            'is_running': False,
            'scanned': 0,
            'total': 0,
            'unusual_found': 0,
            'last_ticker': '',
            'start_time': 0.0
        }

    def get_scan_progress(self) -> Dict[str, Any]:
        """Return current live scan progress snapshot."""
        with self.progress_lock:
            data = dict(self.progress_data)
            if data['is_running'] and data['start_time'] > 0:
                elapsed = time.time() - data['start_time']
                data['elapsed_sec'] = int(elapsed)
                pct = (data['scanned'] / data['total']) if data['total'] > 0 else 0.0
                data['pct'] = round(pct * 100, 1)
                if pct > 0.05:
                    est_total = elapsed / pct
                    data['est_remaining_sec'] = max(0, int(est_total - elapsed))
                else:
                    data['est_remaining_sec'] = 0
            else:
                data['elapsed_sec'] = 0
                data['est_remaining_sec'] = 0
                data['pct'] = 0.0
            return data

    def _update_progress(self, ticker_clean: str):
        """Thread-safe progress update."""
        with self.progress_lock:
            self.progress_data['scanned'] += 1
            self.progress_data['last_ticker'] = ticker_clean

    def _increment_unusual(self):
        """Thread-safe unusual counter increment."""
        with self.progress_lock:
            self.progress_data['unusual_found'] += 1

    def _evaluate_single_ticker(
        self,
        ticker: str,
        as_of_date: Optional[str],
        lookback_days: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single ticker for unusual activity.
        Returns result dict or None if ticker should be skipped.
        This method is thread-safe for parallel execution.
        """
        ticker_clean = ticker.upper().strip()
        if not ticker_clean.endswith('.JK'):
            ticker_clean = f"{ticker_clean}.JK"

        self._update_progress(ticker_clean)

        # 1. Load from DB
        df = self.db.get_prices_with_indicators(ticker_clean, end=as_of_date)

        if df.empty or len(df) < lookback_days:
            return None

        # Ensure turnover_5d calculation on-the-fly
        if 'turnover_5d' not in df.columns:
            df['turnover_5d'] = (df['Close'] * df['Volume']).rolling(5, min_periods=3).mean()

        # Evaluate observation tags
        tags = evaluate_observation_tags(df, lookback_days=lookback_days)
        unusual_tags = [t for t in tags if t.get('is_unusual')]

        if not unusual_tags:
            return None

        self._increment_unusual()

        latest_row  = df.iloc[-1]
        date_str    = df.index[-1].strftime('%Y-%m-%d')
        close_p     = float(latest_row['Close'])
        turnover_5d = float(latest_row['turnover_5d']) if pd.notna(latest_row['turnover_5d']) else None

        # Add Rarity and Streak details to each tag
        processed_tags = []
        for tag in tags:
            t_id = tag['tag_id']
            t_dict = dict(tag)

            # Add streak for trend alignment or persistent unusual tags
            streak_val = get_condition_streak(df, t_id)
            if streak_val > 1:
                t_dict['description'] += f" (berlangsung {streak_val} hari berturut-turut)"

            # Add Rarity Context for unusual tags
            if tag.get('is_unusual'):
                rarity_info = get_rarity_context(ticker_clean, t_id, df_history=df, lookback_days=252)
                t_dict['rarity_text'] = rarity_info.get('summary_text', '')

            processed_tags.append(t_dict)

        liquidity_note = get_liquidity_note(turnover_5d)

        return {
            'ticker': ticker_clean,
            'date': date_str,
            'close': close_p,
            'turnover_5d': turnover_5d,
            'liquidity_note': liquidity_note,
            'unusual_count': len(unusual_tags),
            'all_tags': processed_tags,
            'unusual_tags': [t for t in processed_tags if t.get('is_unusual')]
        }

    def scan_unusual_activity(
        self,
        tickers: Optional[List[str]] = None,
        as_of_date: Optional[str] = None,
        lookback_days: int = 60,
        max_results: int = 15,
        max_new_fetches_per_run: int = 25,
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Scan stock universe for stocks showing unusual technical activity relative to their own history.
        Uses Cache-First Strategy with a safety cap on new yfinance fetches to prevent Gunicorn timeouts.

        Args:
            parallel: Use ThreadPoolExecutor for parallel evaluation (default: True).
        """
        if tickers is None:
            db_tickers = self.db.get_tickers()
            master_universe = fetch_live_idx_tickers()

            # Combine cached tickers with master universe list
            combined = list(dict.fromkeys(db_tickers + master_universe))
            tickers = combined

        total_universe = len(tickers)

        with self.progress_lock:
            self.progress_data['is_running'] = True
            self.progress_data['scanned'] = 0
            self.progress_data['total'] = total_universe
            self.progress_data['unusual_found'] = 0
            self.progress_data['last_ticker'] = ''
            self.progress_data['start_time'] = time.time()

        scanned_results = []

        try:
            if parallel and total_universe > 50:
                # Parallel evaluation for large universes
                scanned_results = self._scan_parallel(
                    tickers, as_of_date, lookback_days,
                    max_new_fetches_per_run
                )
            else:
                # Sequential evaluation for small batches (specific tickers)
                scanned_results = self._scan_sequential(
                    tickers, as_of_date, lookback_days,
                    max_new_fetches_per_run
                )

            # Rank by count of unusual conditions met (descending)
            scanned_results.sort(key=lambda x: x['unusual_count'], reverse=True)
            top_results = scanned_results[:max_results]

            # Attach Market Breadth & Sector Context after top_results are determined
            for item in top_results:
                t = item['ticker']

                # Market Breadth context for primary unusual condition
                first_unusual_id = item['unusual_tags'][0]['tag_id'] if item['unusual_tags'] else ''
                item['breadth_context'] = get_market_breadth_context(
                    first_unusual_id, top_results, total_universe_count=total_universe
                )

                # Sector context (informative only)
                item['sector_context'] = get_sector_context(t, top_results)

            return top_results
        finally:
            with self.progress_lock:
                self.progress_data['is_running'] = False

    def _scan_sequential(
        self,
        tickers: List[str],
        as_of_date: Optional[str],
        lookback_days: int,
        max_new_fetches_per_run: int,
    ) -> List[Dict[str, Any]]:
        """Sequential scan — used for small batches or when parallel=False."""
        results = []
        new_fetches_count = 0

        for ticker in tickers:
            ticker_clean = ticker.upper().strip()
            if not ticker_clean.endswith('.JK'):
                ticker_clean = f"{ticker_clean}.JK"

            with self.progress_lock:
                self.progress_data['scanned'] += 1
                self.progress_data['last_ticker'] = ticker_clean

            # 1. Load from DB
            df = self.db.get_prices_with_indicators(ticker_clean, end=as_of_date)

            # 2. Auto-fetch fallback if DB is empty / missing rows (Capped per run)
            if df.empty or len(df) < lookback_days:
                if new_fetches_count >= max_new_fetches_per_run:
                    continue
                try:
                    df_prices = self.provider.get_or_fetch(ticker_clean, period_days=252*2)
                    new_fetches_count += 1
                    if not df_prices.empty:
                        df_ind = compute_indicators(df_prices)
                        self.db.save_indicators(ticker_clean, df_ind)
                        df = self.db.get_prices_with_indicators(ticker_clean, end=as_of_date)
                except Exception as exc:
                    print(f"[Scanner] Warning: Failed auto-fetch for {ticker_clean}: {exc}")
                    continue

            if df.empty or len(df) < lookback_days:
                continue

            # Ensure turnover_5d calculation on-the-fly
            if 'turnover_5d' not in df.columns:
                df['turnover_5d'] = (df['Close'] * df['Volume']).rolling(5, min_periods=3).mean()

            # Evaluate observation tags
            tags = evaluate_observation_tags(df, lookback_days=lookback_days)
            unusual_tags = [t for t in tags if t.get('is_unusual')]

            if not unusual_tags:
                continue

            with self.progress_lock:
                self.progress_data['unusual_found'] += 1

            latest_row  = df.iloc[-1]
            date_str    = df.index[-1].strftime('%Y-%m-%d')
            close_p     = float(latest_row['Close'])
            turnover_5d = float(latest_row['turnover_5d']) if pd.notna(latest_row['turnover_5d']) else None

            # Add Rarity and Streak details to each tag
            processed_tags = []
            for tag in tags:
                t_id = tag['tag_id']
                t_dict = dict(tag)

                streak_val = get_condition_streak(df, t_id)
                if streak_val > 1:
                    t_dict['description'] += f" (berlangsung {streak_val} hari berturut-turut)"

                if tag.get('is_unusual'):
                    rarity_info = get_rarity_context(ticker_clean, t_id, df_history=df, lookback_days=252)
                    t_dict['rarity_text'] = rarity_info.get('summary_text', '')

                processed_tags.append(t_dict)

            liquidity_note = get_liquidity_note(turnover_5d)

            results.append({
                'ticker': ticker_clean,
                'date': date_str,
                'close': close_p,
                'turnover_5d': turnover_5d,
                'liquidity_note': liquidity_note,
                'unusual_count': len(unusual_tags),
                'all_tags': processed_tags,
                'unusual_tags': [t for t in processed_tags if t.get('is_unusual')]
            })

        return results

    def _scan_parallel(
        self,
        tickers: List[str],
        as_of_date: Optional[str],
        lookback_days: int,
        max_new_fetches_per_run: int,
    ) -> List[Dict[str, Any]]:
        """
        Parallel scan using ThreadPoolExecutor for faster evaluation.
        Note: yfinance fetch is NOT parallelized (rate limit), only DB reads + evaluation.
        """
        results = []
        workers = min(MAX_EVAL_WORKERS, len(tickers))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_ticker = {}
            for ticker in tickers:
                future = executor.submit(
                    self._evaluate_single_ticker,
                    ticker, as_of_date, lookback_days,
                )
                future_to_ticker[future] = ticker

            for future in as_completed(future_to_ticker):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    ticker = future_to_ticker[future]
                    print(f"[Scanner] Error evaluating {ticker}: {exc}")

        return results

    def format_telegram_report(self, scanned_results: List[Dict[str, Any]], as_of_date: Optional[str] = None) -> str:
        """
        Format scan results into clean, descriptive Telegram markdown message.
        """
        if not scanned_results:
            return f"📋 **Scan Aktivitas Harian**\n\nTidak ditemukan saham dengan aktivitas di luar kebiasaan harian.\n\n{MANDATORY_DISCLAIMER}"

        date_hdr = as_of_date or scanned_results[0]['date']
        lines = [
            f"📋 **Scan Observasi Teknikal Harian ({date_hdr})**",
            f"_{len(scanned_results)} saham dengan aktivitas di luar kebiasaan histori 60 hari_\n"
        ]

        for i, item in enumerate(scanned_results, 1):
            ticker = item['ticker'].replace('.JK', '')
            close  = item['close']
            liq    = item['liquidity_note']
            lines.append(f"**{i}. {ticker}** (Rp {close:,.0f}) — _{liq}_")

            for tag in item['all_tags']:
                icon = "🔥" if tag.get('is_unusual') else "•"
                lines.append(f"   {icon} {tag['description']}")
                if tag.get('rarity_text'):
                    lines.append(f"      └─ {tag['rarity_text']}")

            if item.get('breadth_context'):
                lines.append(f"   🌐 {item['breadth_context']}")
            if item.get('sector_context'):
                lines.append(f"   🏢 {item['sector_context']}")

            lines.append("")

        lines.append(f"---\n{MANDATORY_DISCLAIMER}")
        return "\n".join(lines)

    def format_cli_report(self, scanned_results: List[Dict[str, Any]], as_of_date: Optional[str] = None) -> str:
        """
        Format scan results for CLI output.
        """
        if not scanned_results:
            return "Tidak ditemukan saham meyakinkan dengan aktivitas teknikal di luar kebiasaan."

        date_hdr = as_of_date or scanned_results[0]['date']
        lines = [
            "="*70,
            f"  SCAN OBSERVASI TEKNIKAL HARIAN — {date_hdr}",
            "  (Daftar Saham Beraktivitas di Luar Kebiasaan Histori 60-Hari)",
            "="*70
        ]

        for i, item in enumerate(scanned_results, 1):
            ticker = item['ticker']
            close  = item['close']
            lines.append(f"\n  {i:2d}. {ticker:<10} Rp {close:,.0f} ({item['unusual_count']} kondisi unusual)")
            lines.append(f"      [{item['liquidity_note']}]")

            for tag in item['all_tags']:
                prefix = "    [!] " if tag.get('is_unusual') else "    [-] "
                lines.append(f"{prefix}{tag['description']}")
                if tag.get('rarity_text'):
                    lines.append(f"        └─ {tag['rarity_text']}")

            if item.get('breadth_context'):
                lines.append(f"    [Market] {item['breadth_context']}")
            if item.get('sector_context'):
                lines.append(f"    [Sector] {item['sector_context']}")

        lines.append("\n" + "="*70)
        lines.append("  PERINGATAN WAJIB:")
        lines.append(f"  {MANDATORY_DISCLAIMER.replace(chr(10), chr(10) + '  ')}")
        lines.append("="*70)
        return "\n".join(lines)


if __name__ == '__main__':
    db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    scanner = TechnicalObservationScanner(db)

    print("Running Enhanced Technical Observation Scanner across ALL Live BEI Tickers...")
    results = scanner.scan_unusual_activity(max_results=5)
    print(scanner.format_cli_report(results))
