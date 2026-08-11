"""
cli.py — CLI Observasi Teknikal Harian untuk IDX-Screener v3 (Descriptive, Non-Predictive)

MODIFIKASI V3:
- Menggantikan skor total 0-100 & sinyal BULLISH/BEARISH/BUY/SELL dengan TechnicalObservationScanner.
- Output murni deskriptif (fakta teknikal di luar kebiasaan relatif 60-hari saham).
- Terpisah dari strategi Dividend Drift yang divalidasi out-of-sample.

Cara pakai:
    # Scan seluruh universe saham di DB
    python -m src.scoring.cli

    # Scan saham tertentu
    python -m src.scoring.cli BBCA ASII TLKM BMRI

    # Format Telegram preview
    python -m src.scoring.cli --telegram
"""

import argparse
import os
import sys
import io
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import config as app_config
from data.database import DatabaseManager
from scoring.scanner import TechnicalObservationScanner


def main():
    parser = argparse.ArgumentParser(
        description='IDX-Screener v3 — Daily Technical Observation Scanner (Descriptive)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python -m src.scoring.cli
  python -m src.scoring.cli BBCA ASII TLKM BMRI
  python -m src.scoring.cli --telegram
        """
    )
    parser.add_argument(
        'tickers', nargs='*', default=None,
        help='Kode saham opsional (tanpa ticker = scan seluruh database)'
    )
    parser.add_argument(
        '--db', type=str, default=None,
        help='Path ke SQLite database (default: data/idx_screener.db)'
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='Tanggal scan opsional (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--max-results', type=int, default=10,
        help='Maksimum jumlah saham unusual yang ditampilkan (default: 10)'
    )
    parser.add_argument(
        '--telegram', action='store_true',
        help='Tampilkan format Telegram Markdown'
    )

    args = parser.parse_args()

    db_path = args.db or os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    scanner = TechnicalObservationScanner(db)

    results = scanner.scan_unusual_activity(
        tickers=args.tickers if args.tickers else None,
        as_of_date=args.date,
        max_results=args.max_results
    )

    if args.telegram:
        print(scanner.format_telegram_report(results, as_of_date=args.date))
    else:
        print(scanner.format_cli_report(results, as_of_date=args.date))


if __name__ == '__main__':
    main()
