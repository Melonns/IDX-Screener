"""
scanner.py — Daily Technical Observation Scanner (Descriptive, Non-Predictive)

Prinsip Utama:
- Deskriptif, bukan prediktif. Tidak ada skor total 0-100, tidak ada sinyal BULLISH/BEARISH/BUY/SELL.
- Menyaring saham berdasarkan Unusual Activity (aktivitas di luar kebiasaan relatif terhadap histori 60 hari saham itu sendiri).
- Ranking berdasarkan JUMLAH kondisi unusual yang terpenuhi bersamaan.
- Auto-fallback ke yfinance jika data di SQLite lokal kosong (misal saat berjalan di Replit).
- Terpisah secara eksplisit dari strategi yang sudah divalidasi (seperti Dividend Drift).
"""

import os, sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
from data.provider import YFinanceProvider
from data.ingestion import compute_indicators
from scoring.observation_tags import evaluate_observation_tags, get_liquidity_note
from scoring.rarity_context import get_rarity_context, get_condition_streak
from scoring.market_breadth import get_market_breadth_context, get_sector_context
import config as app_config

MANDATORY_DISCLAIMER = """
⚠️  Ringkasan aktivitas teknikal harian, BUKAN sinyal beli/jual.
Riset internal membuktikan indikator teknikal murni tidak memprediksi arah harga ke depan.
Gunakan sebagai titik awal riset manual (berita, laporan keuangan, kondisi sektor).
""".strip()


class TechnicalObservationScanner:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.provider = YFinanceProvider(db)

    def scan_unusual_activity(
        self,
        tickers: Optional[List[str]] = None,
        as_of_date: Optional[str] = None,
        lookback_days: int = 60,
        max_results: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Scan stock universe for stocks showing unusual technical activity relative to their own history.
        If data is missing from local SQLite (e.g. freshly cloned Replit), automatically fetches from yfinance.
        """
        if tickers is None:
            tickers = self.db.get_tickers()
            if not tickers: # Default universe fallback if DB is brand new
                tickers = ['BBRI.JK', 'BBCA.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK',
                           'ANTM.JK', 'ADRO.JK', 'PTBA.JK', 'CTRA.JK', 'UNVR.JK',
                           'HMSP.JK', 'ICBP.JK', 'ISAT.JK', 'ACES.JK', 'GGRM.JK']

        scanned_results = []
        total_universe = len(tickers)

        for ticker in tickers:
            ticker_clean = ticker.upper().strip()
            if not ticker_clean.endswith('.JK'):
                ticker_clean = f"{ticker_clean}.JK"

            # 1. Load from DB
            df = self.db.get_prices_with_indicators(ticker_clean, end=as_of_date)

            # 2. Auto-fetch fallback if DB is empty / missing rows
            if df.empty or len(df) < lookback_days:
                try:
                    df_prices = self.provider.get_or_fetch(ticker_clean, period_days=252*2)
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

            scanned_results.append({
                'ticker': ticker_clean,
                'date': date_str,
                'close': close_p,
                'turnover_5d': turnover_5d,
                'liquidity_note': liquidity_note,
                'unusual_count': len(unusual_tags),
                'all_tags': processed_tags,
                'unusual_tags': [t for t in processed_tags if t.get('is_unusual')]
            })

        # Rank by count of unusual conditions met (descending)
        scanned_results.sort(key=lambda x: x['unusual_count'], reverse=True)
        top_results = scanned_results[:max_results]

        # Attach Market Breadth & Sector Context after top_results are determined
        for item in top_results:
            t = item['ticker']
            
            # Market Breadth context for primary unusual condition
            first_unusual_id = item['unusual_tags'][0]['tag_id'] if item['unusual_tags'] else ''
            item['breadth_context'] = get_market_breadth_context(first_unusual_id, top_results, total_universe_count=total_universe)
            
            # Sector context (informative only)
            item['sector_context']  = get_sector_context(t, top_results)

        return top_results

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
            return "Tidak ditemukan saham dengan aktivitas teknikal di luar kebiasaan."

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

    print("Running Enhanced Technical Observation Scanner...")
    results = scanner.scan_unusual_activity(max_results=5)
    print(scanner.format_cli_report(results))
