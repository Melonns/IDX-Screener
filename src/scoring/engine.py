"""
engine.py — ScoringEngine untuk IDX-Screener v2.

Core logic: ambil OHLCV + indicators DataFrame, jalankan semua scorer,
gabungkan jadi skor 0–100 dengan breakdown JSON per komponen.

Alur:
    ScoringEngine.score(ticker, df)
        → jalankan setiap Indicator.compute(df)
        → sum semua skor
        → tentukan label (BULLISH/BEARISH/NEUTRAL)
        → return dict lengkap dengan breakdown
"""

from datetime import date
from typing import Optional

import pandas as pd

from data.database import DatabaseManager
from data.provider import YFinanceProvider
from data.ingestion import compute_indicators
from features.indicators import ALL_SCORERS, Indicator
from risk.manager import RiskManager
from scoring.config import SCORING_CONFIG


class ScoringEngine:
    """
    Quantitative scoring engine berbasis rule-based weighted indicators.

    Output utama: skor 0–100 dengan breakdown per komponen yang bisa diaudit.

    PENTING:
    - Skor tinggi bukan jaminan profit — ini alat bantu screening.
    - Selalu validasi backtest sebelum menggunakan sinyal untuk trading nyata.
    - CLI output yang bagus ≠ bukti sinyal predictive. Backtest adalah pembuktian.

    Args:
        scorers: List Indicator yang dipakai. Default: ALL_SCORERS (7 indikator, total max 100).
        config: Dict konfigurasi threshold. Default: SCORING_CONFIG.
    """

    def __init__(
        self,
        scorers: list[Indicator] = None,
        config: dict = None,
    ) -> None:
        self.scorers = scorers if scorers is not None else ALL_SCORERS
        self.config  = config if config is not None else SCORING_CONFIG

        # Validasi: pastikan total max score = 100
        total_max = sum(s.max_score for s in self.scorers)
        if total_max != 100:
            import warnings
            warnings.warn(
                f"Total max score = {total_max}, bukan 100. "
                f"Skor akhir tidak akan dalam skala 0-100 yang proper.",
                stacklevel=2,
            )

    def score(
        self,
        ticker: str,
        df: pd.DataFrame,
        today: str = None,
    ) -> dict:
        """
        Hitung skor untuk satu ticker dari DataFrame OHLCV + indicators.

        Args:
            ticker: Kode saham (misal 'BBCA.JK')
            df: DataFrame dengan kolom OHLCV + indikator teknikal.
                Bisa berisi full history — hanya baris terakhir yang di-score.
            today: Tanggal sinyal (default: tanggal dari row terakhir df)

        Returns:
            Dict hasil scoring:
            {
                'kode': str,
                'tanggal': str,
                'skor_total': int,   # 0-100
                'sinyal': str,       # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
                'scoring_version': str,
                'breakdown': list[dict],   # per indikator
                'risk': dict | None,       # stop loss, position size
            }
        """
        if df.empty:
            return self._empty_result(ticker, today or date.today().isoformat())

        # Tanggal sinyal = tanggal row terakhir
        tanggal = today or str(df.index[-1])[:10]

        breakdown = []
        total_score = 0

        for scorer in self.scorers:
            try:
                result = scorer.compute(df)
                breakdown.append(result)
                total_score += result.get('skor', 0)
            except Exception as exc:
                breakdown.append({
                    'indikator': scorer.name,
                    'nilai': f"Error: {exc}",
                    'kontribusi': "Tidak dapat dihitung",
                    'skor': 0,
                    'maks': scorer.max_score,
                })

        # Clamp ke [0, 100]
        total_score = max(0, min(total_score, 100))

        # Tentukan label sinyal
        bullish_threshold = self.config['bullish_threshold']
        bearish_threshold = self.config['bearish_threshold']

        if total_score >= bullish_threshold:
            sinyal = 'BULLISH'
        elif total_score <= bearish_threshold:
            sinyal = 'BEARISH'
        else:
            sinyal = 'NEUTRAL'

        # Hitung risk management
        risk = None
        try:
            rm = RiskManager()
            risk = rm.calculate(df)
        except Exception:
            pass

        return {
            'kode': ticker,
            'tanggal': tanggal,
            'skor_total': total_score,
            'sinyal': sinyal,
            'scoring_version': self.config.get('scoring_version', 'rule_v1.0'),
            'breakdown': breakdown,
            'risk': risk,
        }

    def score_from_db(
        self,
        ticker: str,
        db: DatabaseManager,
        lookback_days: int = 100,
        save_to_db: bool = True,
    ) -> dict:
        """
        Score ticker menggunakan data dari database (via JOIN prices + indicators).
        Lebih efisien dari score() karena tidak perlu fetch dari yfinance.

        Args:
            ticker: Kode saham
            db: DatabaseManager instance
            lookback_days: Berapa hari ke belakang yang di-load
            save_to_db: Kalau True, simpan hasil ke tabel signals

        Returns:
            Dict hasil scoring (sama seperti score()).
        """
        from datetime import timedelta
        start = (date.today() - timedelta(days=lookback_days)).isoformat()
        df = db.get_prices_with_indicators(ticker, start=start)

        if df.empty:
            return self._empty_result(ticker, date.today().isoformat())

        result = self.score(ticker, df)

        if save_to_db and result['skor_total'] > 0:
            try:
                db.save_signal(result)
            except Exception as exc:
                print(f"[ScoringEngine] Warning: Gagal simpan sinyal ke DB: {exc}")

        return result

    def score_batch(
        self,
        tickers: list[str],
        db: DatabaseManager,
        min_score: int = 0,
        save_to_db: bool = True,
    ) -> list[dict]:
        """
        Score beberapa ticker sekaligus dari database.
        Return hanya sinyal di atas min_score, sorted descending.

        Args:
            tickers: List kode saham
            db: DatabaseManager
            min_score: Filter minimum skor (default 0 = return semua)
            save_to_db: Simpan hasil ke tabel signals

        Returns:
            List dict hasil scoring, sorted by skor_total descending.
        """
        results = []

        for ticker in tickers:
            try:
                result = self.score_from_db(ticker, db, save_to_db=save_to_db)
                if result['skor_total'] >= min_score:
                    results.append(result)
            except Exception as exc:
                print(f"[ScoringEngine] Error scoring {ticker}: {exc}")

        # Sort by score descending
        results.sort(key=lambda x: x['skor_total'], reverse=True)
        return results

    def _empty_result(self, ticker: str, tanggal: str) -> dict:
        return {
            'kode': ticker,
            'tanggal': tanggal,
            'skor_total': 0,
            'sinyal': 'NEUTRAL',
            'scoring_version': self.config.get('scoring_version', 'rule_v1.0'),
            'breakdown': [],
            'risk': None,
        }
