"""
cli.py — CLI test lokal untuk ScoringEngine IDX-Screener v2.

PENTING: Output CLI yang bagus BUKAN bukti sinyal predictive.
Ini hanya validasi bahwa logic berjalan tanpa error.
Pembuktian sesungguhnya ada di backtest (src/backtest/engine.py).

Cara pakai:
    # Score satu saham
    python -m src.scoring.cli BBCA

    # Score beberapa saham
    python -m src.scoring.cli BBCA ASII TLKM BMRI

    # Pakai database lokal
    python -m src.scoring.cli BBCA --db data/idx_screener.db

    # Fetch langsung dari yfinance tanpa database
    python -m src.scoring.cli BBCA --live

    # Filter hanya yang BULLISH dengan skor minimum
    python -m src.scoring.cli BBCA ASII TLKM --min-score 60
"""

import argparse
import os
import sys
from pathlib import Path

# ─── Fix Windows terminal Unicode encoding ───────────────────────────────────
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import config as app_config
from data.database import DatabaseManager
from data.provider import YFinanceProvider
from data.ingestion import compute_indicators
from scoring.engine import ScoringEngine
from scoring.config import SCORING_CONFIG


# ─── Formatting helpers ───────────────────────────────────────────────────────

SINYAL_PREFIX = {
    'BULLISH': '[BULLISH]',
    'BEARISH': '[BEARISH]',
    'NEUTRAL': '[NEUTRAL]',
}

SINYAL_ICON = {
    'BULLISH': '(+)',
    'BEARISH': '(-)',
    'NEUTRAL': '(=)',
}


def format_score_bar(skor: int, width: int = 20) -> str:
    """Visual progress bar untuk skor."""
    filled = int(skor / 100 * width)
    bar = '#' * filled + '.' * (width - filled)
    return f"[{bar}] {skor}/100"


def format_result(result: dict, verbose: bool = False) -> str:
    """Format hasil scoring jadi string yang enak dibaca di terminal."""
    ticker  = result['kode']
    tanggal = result['tanggal']
    skor    = result['skor_total']
    sinyal  = result['sinyal']
    icon    = SINYAL_ICON.get(sinyal, '( )')
    label   = SINYAL_PREFIX.get(sinyal, sinyal)
    version = result.get('scoring_version', '')

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {icon} {ticker:<10} Skor: {format_score_bar(skor)}  {label}")
    lines.append(f"  Tanggal: {tanggal}   Versi: {version}")
    lines.append(f"{'='*60}")

    # Breakdown
    breakdown = result.get('breakdown', [])
    if breakdown:
        lines.append("  Breakdown:")
        for item in breakdown:
            nama        = item.get('indikator', '?')
            skor_item   = item.get('skor', 0)
            maks_item   = item.get('maks', 0)
            nilai       = item.get('nilai', '')
            kontribusi  = item.get('kontribusi', '')

            # Bar kecil per indikator
            ratio = skor_item / maks_item if maks_item > 0 else 0
            mini_filled = int(ratio * 8)
            mini_bar = '|' * mini_filled + '.' * (8 - mini_filled)

            lines.append(f"    {nama:<22} [{mini_bar}] {skor_item:>2}/{maks_item}")
            if verbose:
                lines.append(f"      Nilai     : {nilai}")
                lines.append(f"      Kontribusi: {kontribusi}")
            else:
                lines.append(f"      -> {kontribusi}")

    # Risk management
    risk = result.get('risk')
    if risk and risk.get('stop_loss'):
        lines.append(f"\n  [Risk Management]")
        lines.append(f"    Stop Loss    : {risk['stop_loss']:,.0f} (-{risk.get('risk_pct', 0):.1f}%)")
        if risk.get('position_pct'):
            lines.append(f"    Max Position : {risk['position_pct']:.0f}% kapital (asumsi risk 1%)")
        if not risk.get('liquidity', True):
            lines.append(f"    PERINGATAN: {risk.get('liquidity_warning', '')}")

    lines.append(f"{'='*60}")
    lines.append("  [!] Ini alat bantu screening, bukan rekomendasi investasi.")
    lines.append("      Selalu lakukan analisis manual sebelum keputusan trading.")

    return '\n'.join(lines)


# ─── Main logic ───────────────────────────────────────────────────────────────

def run_cli(
    tickers: list[str],
    db_path: str = None,
    live: bool = False,
    min_score: int = 0,
    verbose: bool = False,
    period_days: int = 100,
) -> None:
    """
    Jalankan scoring untuk daftar ticker dan print ke terminal.

    Args:
        tickers: List kode saham
        db_path: Path SQLite (None = pakai default dari config)
        live: Kalau True, fetch langsung dari yfinance (bypass DB)
        min_score: Filter minimum skor
        verbose: Print detail nilai per indikator
        period_days: Berapa hari data yang di-load dari DB (atau di-fetch)
    """
    print(f"\n{'='*60}")
    print(f"  IDX-Screener v2 -- Quantitative Scoring System")
    print(f"  Mode: {'Live (yfinance)' if live else 'Database'}")
    print(f"{'='*60}")
    print(f"")
    print(f"  [!] REMINDER: Output CLI bukan bukti sinyal predictive.")
    print(f"      Jalankan backtest (src/backtest/engine.py) untuk validasi.")

    if db_path is None:
        db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')

    db = DatabaseManager(db_path)
    engine = ScoringEngine()

    results = []
    failed  = []

    for ticker in tickers:
        ticker = ticker.upper().strip()
        if not ticker.endswith('.JK'):
            ticker = f'{ticker}.JK'

        try:
            if live:
                # Fetch langsung dari yfinance, simpan ke DB
                provider = YFinanceProvider(db, request_delay=1.0)
                df_prices = provider.get_or_fetch(ticker, period_days=period_days)

                if df_prices.empty:
                    failed.append((ticker, "Tidak ada data dari yfinance"))
                    continue

                # Hitung indikator on-the-fly
                df_indicators = compute_indicators(df_prices)
                if not df_indicators.empty:
                    df = df_prices.join(df_indicators, how='left')
                else:
                    df = df_prices

                result = engine.score(ticker, df)

            else:
                # Ambil dari database
                result = engine.score_from_db(
                    ticker, db,
                    lookback_days=period_days,
                    save_to_db=False,
                )

            if result['skor_total'] >= min_score:
                results.append(result)
            else:
                print(f"  Skip: {ticker}: Skor {result['skor_total']} < min_score {min_score}, di-skip.")

        except Exception as exc:
            failed.append((ticker, str(exc)))
            print(f"\n  ✗ Error scoring {ticker}: {exc}")

    # Print results sorted by score
    results.sort(key=lambda x: x['skor_total'], reverse=True)
    for result in results:
        print(format_result(result, verbose=verbose))

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(results)} sinyal, {len(failed)} gagal")
    if results:
        bullish_count = sum(1 for r in results if r['sinyal'] == 'BULLISH')
        print(f"  Bullish  : {bullish_count}")
        print(f"  Neutral  : {sum(1 for r in results if r['sinyal'] == 'NEUTRAL')}")
        print(f"  Bearish  : {sum(1 for r in results if r['sinyal'] == 'BEARISH')}")
        top = results[0]
        print(f"  Top Pick : {top['kode']} -- Skor {top['skor_total']} [{top['sinyal']}]")
    if failed:
        print(f"  Yang gagal:")
        for ticker, reason in failed:
            print(f"    - {ticker}: {reason[:70]}")
    print(f"{'='*60}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IDX-Screener v2 — Local Scoring CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python -m src.scoring.cli BBCA
  python -m src.scoring.cli BBCA ASII TLKM BMRI --verbose
  python -m src.scoring.cli BBCA --live              (fetch dari yfinance)
  python -m src.scoring.cli BBCA ASII --min-score 60
        """
    )
    parser.add_argument(
        'tickers', nargs='+',
        help='Kode saham (tanpa .JK juga oke, misal: BBCA ASII TLKM)'
    )
    parser.add_argument(
        '--db', type=str, default=None,
        help='Path ke SQLite database (default: data/idx_screener.db)'
    )
    parser.add_argument(
        '--live', action='store_true',
        help='Fetch langsung dari yfinance (tidak perlu data di DB dulu)'
    )
    parser.add_argument(
        '--min-score', type=int, default=0,
        help='Filter: hanya tampilkan skor >= ini (default: 0 = tampilkan semua)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print nilai detail per indikator'
    )
    parser.add_argument(
        '--period', type=int, default=100,
        help='Berapa hari data yang di-load (default: 100)'
    )

    args = parser.parse_args()

    run_cli(
        tickers=args.tickers,
        db_path=args.db,
        live=args.live,
        min_score=args.min_score,
        verbose=args.verbose,
        period_days=args.period,
    )


if __name__ == '__main__':
    main()
