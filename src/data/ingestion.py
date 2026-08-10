"""
ingestion.py — Bulk download + compute daily_indicators untuk IDX-Screener v2.

Cara pakai:
    # Download 10 saham starter universe (3 tahun)
    python -m src.data.ingestion

    # Download saham tertentu
    python -m src.data.ingestion --tickers BBCA ASII TLKM BMRI

    # Force refresh semua data
    python -m src.data.ingestion --force-refresh

    # Hitung ulang indikator dari data yang sudah ada (tanpa fetch)
    python -m src.data.ingestion --indicators-only --tickers BBCA

Setelah download, script ini otomatis menghitung dan menyimpan
daily_indicators (RSI, EMA, MACD, BB, ATR, RVOL) untuk semua ticker.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

# ─── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import config
from data.database import DatabaseManager
from data.provider import YFinanceProvider

# ─── Starter universe (diverse, sesuai review guidelines) ─────────────────────
# Pilih berdasarkan karakteristik berbeda:
# - Blue chip high-liquidity: data bersih, spread kecil
# - Mid-cap: more volatile, representatif
# - Sektoral berbeda: bank, telco, mining, consumer
STARTER_UNIVERSE = [
    # Blue chip high-liquidity (perbankan)
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK',
    # Mid-cap, industri/infrastruktur
    'ASII.JK', 'TLKM.JK', 'PGAS.JK',
    # Consumer goods
    'UNVR.JK',
    # Mining/energy
    'INDY.JK', 'ADRO.JK',
    # Property/konstruksi (lebih volatile)
    'SMGR.JK',
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung semua technical indicators dari OHLCV DataFrame.
    Return DataFrame dengan kolom indikator (bukan raw OHLCV).

    Input DataFrame harus punya kolom: Open, High, Low, Close, Volume
    dengan DatetimeIndex.

    Kolom output:
        rsi_14, ema_9, ema_21, ema_50,
        macd, macd_signal, macd_diff,
        bb_upper, bb_lower, bb_width,
        atr_14, volume_ratio_20d
    """
    if df.empty or len(df) < 50:
        print(f"  Warning: Data terlalu sedikit ({len(df)} baris), minimal 50 baris untuk hitung semua indikator.")
        return pd.DataFrame()

    result = pd.DataFrame(index=df.index)

    # ── RSI ──────────────────────────────────────────────────────────────────
    result['rsi_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # ── EMA ──────────────────────────────────────────────────────────────────
    result['ema_9']  = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    result['ema_21'] = EMAIndicator(close=df['Close'], window=21).ema_indicator()
    result['ema_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_obj = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    result['macd']        = macd_obj.macd()
    result['macd_signal'] = macd_obj.macd_signal()
    result['macd_diff']   = macd_obj.macd_diff()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    result['bb_upper'] = bb.bollinger_hband()
    result['bb_lower'] = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    # BB Width = (upper - lower) / mid — deteksi squeeze (low volatility sebelum breakout)
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / bb_mid.replace(0, float('nan'))

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    atr_obj = AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    )
    result['atr_14'] = atr_obj.average_true_range()

    # ── Volume Ratio (RVOL) ───────────────────────────────────────────────────
    # RVOL = volume hari ini / rata-rata volume 20 hari. >2 = spike signifikan.
    vol_ma20 = df['Volume'].rolling(window=20).mean()
    result['volume_ratio_20d'] = df['Volume'] / vol_ma20.replace(0, float('nan'))

    return result


def run_ingestion(
    tickers: list[str],
    period_days: int = 365 * 3,
    force_refresh: bool = False,
    indicators_only: bool = False,
    db_path: str = None,
) -> dict:
    """
    Pipeline lengkap: fetch prices → simpan SQLite → hitung & simpan indicators.

    Args:
        tickers: List kode saham (misal ['BBCA.JK', 'ASII.JK'])
        period_days: Berapa hari historis yang diambil (default 3 tahun)
        force_refresh: Paksa re-fetch meski data sudah ada
        indicators_only: Hanya hitung ulang indikator (skip fetch)
        db_path: Path ke SQLite file (default dari config)

    Returns:
        Dict summary: {'success': [...], 'failed': [...], 'total_rows': int}
    """
    if db_path is None:
        db_path = os.path.join(config.DATA_DIR, 'idx_screener.db')

    db = DatabaseManager(db_path)
    provider = YFinanceProvider(db, request_delay=1.0)

    summary = {'success': [], 'failed': [], 'skipped_indicators': [], 'total_price_rows': 0}
    total = len(tickers)

    print(f"\n{'='*60}")
    print(f"IDX-Screener v2 — Data Ingestion")
    print(f"Tickers  : {total}")
    print(f"Period   : {period_days // 365} tahun ({period_days} hari)")
    print(f"DB Path  : {db_path}")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers, 1):
        ticker = ticker.upper().strip()
        if not ticker.endswith('.JK'):
            ticker = f"{ticker}.JK"

        print(f"\n[{i}/{total}] ── {ticker} ─────────────────────────────")

        # ── Fase 1: Fetch + Simpan Prices ────────────────────────────────────
        if not indicators_only:
            try:
                df_prices = provider.get_or_fetch(
                    ticker,
                    period_days=period_days,
                    force_refresh=force_refresh,
                )
                summary['total_price_rows'] += len(df_prices)
            except Exception as exc:
                print(f"  ✗ Gagal fetch prices: {exc}")
                summary['failed'].append({'ticker': ticker, 'reason': str(exc), 'step': 'fetch'})
                continue
        else:
            # Ambil dari DB langsung
            df_prices = db.get_prices(ticker)
            if df_prices.empty:
                print(f"  ✗ Tidak ada data di database untuk {ticker}. Skip.")
                summary['skipped_indicators'].append(ticker)
                continue

        # ── Fase 2: Hitung + Simpan Indicators ───────────────────────────────
        if df_prices.empty:
            print(f"  ✗ Data prices kosong, skip hitung indikator.")
            summary['failed'].append({'ticker': ticker, 'reason': 'Empty price data', 'step': 'indicators'})
            continue

        try:
            print(f"  Menghitung indikator ({len(df_prices)} baris)...")
            df_indicators = compute_indicators(df_prices)

            if df_indicators.empty:
                print(f"  ✗ Gagal hitung indikator (data terlalu sedikit?).")
                summary['skipped_indicators'].append(ticker)
            else:
                saved = db.save_indicators(ticker, df_indicators)
                print(f"  ✓ {saved} baris indikator disimpan.")
                summary['success'].append(ticker)

        except Exception as exc:
            print(f"  ✗ Gagal hitung indikator: {exc}")
            summary['failed'].append({'ticker': ticker, 'reason': str(exc), 'step': 'indicators'})

    # ── Print Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SELESAI — Ingestion Summary")
    print(f"  Berhasil   : {len(summary['success'])} / {total}")
    print(f"  Gagal      : {len(summary['failed'])}")
    print(f"  Total rows : {summary['total_price_rows']:,}")
    if summary['failed']:
        print(f"\n  Daftar yang gagal:")
        for f in summary['failed']:
            print(f"    - {f['ticker']} ({f['step']}): {f['reason'][:80]}")

    db_stats = db.get_db_stats()
    print(f"\nDatabase Stats:")
    for table, count in db_stats.items():
        print(f"  {table:<25}: {count:,} rows")
    print(f"{'='*60}")

    return summary


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IDX-Screener v2 — Data Ingestion Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # Download starter universe (10 saham, 3 tahun)
  python -m src.data.ingestion

  # Download saham tertentu
  python -m src.data.ingestion --tickers BBCA ASII TLKM

  # Download 5 tahun untuk backtest yang lebih robust
  python -m src.data.ingestion --period 5

  # Hitung ulang indikator (tanpa re-fetch)
  python -m src.data.ingestion --indicators-only

  # Force refresh semua data
  python -m src.data.ingestion --force-refresh
        """
    )
    parser.add_argument(
        '--tickers', nargs='+', default=None,
        help='Daftar ticker (misal: BBCA ASII TLKM). Default: starter universe (10 saham).'
    )
    parser.add_argument(
        '--period', type=int, default=3,
        help='Berapa tahun data historis (default: 3).'
    )
    parser.add_argument(
        '--force-refresh', action='store_true',
        help='Paksa re-fetch meski data sudah ada di database.'
    )
    parser.add_argument(
        '--indicators-only', action='store_true',
        help='Hanya hitung ulang indikator dari data yang sudah ada (skip fetch).'
    )
    parser.add_argument(
        '--db', type=str, default=None,
        help='Path ke SQLite database file (default: data/idx_screener.db).'
    )

    args = parser.parse_args()

    tickers = args.tickers if args.tickers else STARTER_UNIVERSE

    run_ingestion(
        tickers=tickers,
        period_days=args.period * 365,
        force_refresh=args.force_refresh,
        indicators_only=args.indicators_only,
        db_path=args.db,
    )


if __name__ == '__main__':
    main()
