"""
indicators.py — Class-based scorer per indikator teknikal.

Setiap indikator mengimplementasikan interface Indicator dengan method compute()
yang return dict berisi: nilai, interpretasi, skor, dan maks skor.

Output setiap scorer didesain untuk langsung masuk ke ScoringEngine.
Format output:
    {
        'indikator': str,     # nama indikator
        'nilai': str,         # nilai yang terdeteksi (human-readable)
        'kontribusi': str,    # interpretasi sinyal
        'skor': int,          # skor yang diberikan
        'maks': int,          # skor maksimal yang bisa dicapai
    }
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

def _cfg(config: dict | None, key: str):
    """Helper: ambil nilai dari config dict, fallback ke default."""
    if config is not None and key in config:
        return config[key]
    # Import inline to avoid circular import if any
    from scoring.config import SCORING_CONFIG
    return SCORING_CONFIG.get(key, 0.0)


class Indicator(ABC):
    """Interface untuk semua indikator scorer."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nama indikator (untuk display di breakdown)."""
        ...

    @property
    @abstractmethod
    def max_score(self) -> int:
        """Skor maksimal yang bisa dikontribusikan."""
        ...

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> dict:
        """
        Hitung skor dari DataFrame OHLCV + indicators (baris terakhir).

        Args:
            df: DataFrame dengan kolom OHLCV + indikator teknikal.
                Bisa full history atau subset — metode akan pakai row terakhir
                plus context beberapa baris sebelumnya untuk kalkulasi.

        Returns:
            Dict dengan kunci: indikator, nilai, kontribusi, skor, maks.
            Kalau data tidak cukup, return skor=0 dengan keterangan.
        """
        ...

    def _safe_result(self, nilai: str, kontribusi: str, skor: int) -> dict:
        """Helper untuk buat output dict yang konsisten."""
        skor = max(0, min(skor, self.max_score))  # clamp ke [0, max_score]
        return {
            'indikator': self.name,
            'nilai': nilai,
            'kontribusi': kontribusi,
            'skor': skor,
            'maks': self.max_score,
        }

    def _no_data(self, reason: str = "Data tidak cukup") -> dict:
        """Return hasil kosong kalau data tidak tersedia."""
        return self._safe_result(reason, "Tidak dapat dihitung", 0)


# ─────────────────────────────────────────────────────────────────────────────
# RSI Scorer
# ─────────────────────────────────────────────────────────────────────────────

class RSIScorer(Indicator):
    """
    Skor RSI-14 dimodifikasi untuk menangkap sifat Mean-Reverting dari pasar IHSG.
    
    Logika Konseptual:
    - IHSG terbukti mean-reverting (RSI tinggi berkorelasi negatif dengan return N+3).
    - RSI > Overbought: Diberi PENALTI (skor 0 atau negatif jika memungkinkan, tapi dibatasi minimum 0).
    - RSI Oversold Bounce (RSI < Oversold tapi mulai naik): Skor penuh (menangkap pantulan).
    - RSI Mid-zone & Naik: Skor moderat (mengikuti trend tapi belum jenuh).
    - Threshold oversold, overbought, dan mid_zone didelegasikan ke Optuna.
    """

    name = "RSI (Mean-Revert)"
    max_score = 15

    def compute(self, df: pd.DataFrame) -> dict:
        if 'rsi_14' not in df.columns or df['rsi_14'].dropna().empty:
            return self._no_data()

        rsi_series = df['rsi_14'].dropna()
        if len(rsi_series) < 2:
            return self._no_data("RSI perlu minimal 2 data poin")

        rsi_now  = rsi_series.iloc[-1]
        rsi_prev = rsi_series.iloc[-2]
        
        # Threshold ini akan di-tune oleh Optuna
        overbought = _cfg(None, 'rsi_overbought_threshold') # e.g. 70
        oversold   = _cfg(None, 'rsi_oversold_threshold')   # e.g. 30
        mid_zone   = _cfg(None, 'rsi_bullish_threshold')    # e.g. 50

        rsi_display = f"{rsi_now:.1f}"

        # 1. Jenuh Beli (Mean-Reverting Drop Risk)
        if rsi_now >= overbought:
            return self._safe_result(rsi_display, f"Overbought (> {overbought}) — Risiko koreksi tinggi", 0)

        # 2. Oversold Bounce (Sweet spot untuk mean-reversion)
        if rsi_prev <= oversold and rsi_now > rsi_prev:
            return self._safe_result(
                rsi_display,
                f"Oversold Bounce (< {oversold} & naik) — Potensi reversal kuat",
                self.max_score,
            )

        # 3. Mid-zone Momentum
        if mid_zone <= rsi_now < overbought:
            if rsi_now > rsi_prev:
                return self._safe_result(
                    rsi_display,
                    f"Mid-zone momentum menguat",
                    int(self.max_score * 0.6),
                )
            else:
                return self._safe_result(
                    rsi_display,
                    f"Mid-zone tertahan",
                    int(self.max_score * 0.3),
                )

        return self._safe_result(rsi_display, f"RSI lemah/menurun", 0)


# ─────────────────────────────────────────────────────────────────────────────
# EMA Cross Scorer
# ─────────────────────────────────────────────────────────────────────────────

class EMACrossScorer(Indicator):
    """
    Skor berbasis alignment EMA9, EMA21, EMA50 (triple EMA stack).

    Logika:
    - Triple alignment bullish (EMA9 > EMA21 > EMA50): skor penuh
    - EMA9 baru cross EMA21 (golden cross short): skor tinggi
    - EMA9 > EMA21 tapi EMA21 < EMA50: skor parsial (trend masih mixed)
    - EMA bearish alignment: skor 0
    """

    name = "EMA Cross"
    max_score = 20

    def compute(self, df: pd.DataFrame) -> dict:
        needed = {'ema_9', 'ema_21', 'ema_50', 'Close'}
        if not needed.issubset(df.columns) or df['ema_9'].dropna().empty:
            return self._no_data()

        row  = df.dropna(subset=['ema_9', 'ema_21', 'ema_50']).iloc[-1] if len(df.dropna(subset=['ema_9', 'ema_21', 'ema_50'])) > 0 else None
        if row is None:
            return self._no_data()

        ema9  = row['ema_9']
        ema21 = row['ema_21']
        ema50 = row['ema_50']

        display = f"EMA9={ema9:.0f}, EMA21={ema21:.0f}, EMA50={ema50:.0f}"

        # Cek golden cross (EMA9 baru cross EMA21)
        prev_valid = df.dropna(subset=['ema_9', 'ema_21'])
        golden_cross = False
        if len(prev_valid) >= 2:
            prev = prev_valid.iloc[-2]
            if prev['ema_9'] <= prev['ema_21'] and ema9 > ema21:
                golden_cross = True

        # Triple bullish alignment
        if ema9 > ema21 > ema50:
            if golden_cross:
                return self._safe_result(
                    display,
                    "Golden cross + triple bullish alignment (EMA9>EMA21>EMA50)",
                    self.max_score,
                )
            return self._safe_result(display, "Triple bullish alignment (EMA9>EMA21>EMA50)", self.max_score)

        # Partial alignment: EMA9 > EMA21 tapi EMA50 masih di atas
        if ema9 > ema21 and ema21 < ema50:
            return self._safe_result(
                display,
                "EMA9>EMA21 (short bullish) tapi EMA50 masih di atas (trend menengah belum konfirmasi)",
                int(self.max_score * 0.5),
            )

        # EMA9 < EMA21 (bearish short)
        if ema9 < ema21 < ema50:
            return self._safe_result(display, "Triple bearish alignment (EMA9<EMA21<EMA50)", 0)

        # Mixed
        return self._safe_result(display, "EMA alignment mixed — tidak ada sinyal kuat", int(self.max_score * 0.1))


# ─────────────────────────────────────────────────────────────────────────────
# MACD Scorer
# ─────────────────────────────────────────────────────────────────────────────

class MACDScorer(Indicator):
    """
    Skor MACD dimodifikasi menjadi Graded Scoring berbasis momentum histogram yang berkelanjutan.

    Logika Konseptual:
    - Tidak sekadar mengandalkan "cross hari ini" (binary), melainkan menilai seberapa kuat akselerasi trennya.
    - Momentum Akselerasi: Jika histogram (diff) positif dan Tumbuh lebih cepat dari kemarin -> Skor maksimal.
    - Momentum Deselerasi: Jika histogram positif tapi mulai menyusut -> Skor dipotong.
    - Fresh Cross Bonus: Tetap diberikan jika baru saja menyeberang 0 ke atas.
    - Threshold untuk 'berapa lama' fresh cross dihitung didelegasikan ke Optuna.
    """

    name = "MACD (Graded Momentum)"
    max_score = 15

    def compute(self, df: pd.DataFrame) -> dict:
        needed = {'macd', 'macd_signal', 'macd_diff'}
        if not needed.issubset(df.columns):
            return self._no_data()

        valid = df.dropna(subset=['macd_diff'])
        if len(valid) < 3:
            return self._no_data("MACD perlu minimal 3 data poin")

        diff_series = valid['macd_diff']
        diff_now  = diff_series.iloc[-1]
        diff_prev1 = diff_series.iloc[-2]
        diff_prev2 = diff_series.iloc[-3]
        macd_now  = valid['macd'].iloc[-1]

        display = f"Diff={diff_now:.3f}"

        # Jika histogram negatif, tidak ada momentum bullish
        if diff_now <= 0:
            return self._safe_result(display, "Histogram negatif (momentum bearish)", 0)

        cross_window = _cfg(None, 'macd_cross_window') # e.g. 3

        # Hitung berapa hari sejak cross (hari terakhir di mana diff <= 0)
        days_since_cross = 0
        for val in reversed(diff_series.iloc[:-1]):
            if val <= 0:
                break
            days_since_cross += 1

        # Hitung akselerasi (kecepatan pertumbuhan momentum)
        velocity_now = diff_now - diff_prev1
        velocity_prev = diff_prev1 - diff_prev2
        accelerating = velocity_now > velocity_prev and velocity_now > 0

        # Graded Scoring
        if days_since_cross <= cross_window:
            # Fresh Cross Area
            if accelerating:
                return self._safe_result(display, f"Fresh Cross ({days_since_cross} hari) & Momentum Akselerasi", self.max_score)
            else:
                return self._safe_result(display, f"Fresh Cross ({days_since_cross} hari) tapi momentum melambat", int(self.max_score * 0.7))
        else:
            # Mature Trend Area
            if accelerating:
                return self._safe_result(display, f"Trend mature tapi momentum masih akselerasi", int(self.max_score * 0.6))
            else:
                return self._safe_result(display, f"Trend mature & momentum deselerasi (profit taking zone)", int(self.max_score * 0.2))



# ─────────────────────────────────────────────────────────────────────────────
# Volume Scorer (RVOL)
# ─────────────────────────────────────────────────────────────────────────────

class VolumeScorer(Indicator):
    """
    Skor berbasis Relative Volume (RVOL = volume hari ini / avg volume 20 hari).

    Volume spike mengkonfirmasi pergerakan harga. Tanpa volume, breakout lemah.

    Logika:
    - RVOL ≥ spike threshold (default 2.0): skor penuh
    - RVOL ≥ moderate threshold (default 1.5): skor parsial
    - RVOL < 1: volume lemah di bawah rata-rata
    """

    name = "Volume (RVOL)"
    max_score = 20

    def compute(self, df: pd.DataFrame) -> dict:
        if 'volume_ratio_20d' not in df.columns:
            return self._no_data()

        valid = df['volume_ratio_20d'].dropna()
        if valid.empty:
            return self._no_data("RVOL belum bisa dihitung (perlu 20 hari data volume)")

        rvol = valid.iloc[-1]
        spike_threshold    = _cfg(None, 'rvol_spike_threshold')
        moderate_threshold = _cfg(None, 'rvol_moderate_threshold')
        display = f"{rvol:.2f}x rata-rata 20 hari"

        if rvol >= spike_threshold:
            return self._safe_result(display, f"Volume spike signifikan ({rvol:.1f}x) — konfirmasi kuat", self.max_score)

        if rvol >= moderate_threshold:
            ratio = (rvol - moderate_threshold) / (spike_threshold - moderate_threshold)
            skor = int(self.max_score * 0.5 + self.max_score * 0.5 * ratio)
            return self._safe_result(display, f"Volume di atas rata-rata ({rvol:.1f}x) — konfirmasi moderate", skor)

        if rvol >= 1.0:
            return self._safe_result(display, f"Volume rata-rata ({rvol:.1f}x) — tidak ada konfirmasi", int(self.max_score * 0.1))

        return self._safe_result(display, f"Volume di bawah rata-rata ({rvol:.1f}x) — sinyal lemah", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Band Scorer
# ─────────────────────────────────────────────────────────────────────────────

class BollingerScorer(Indicator):
    """
    Skor berbasis posisi harga terhadap Bollinger Bands + deteksi squeeze.

    Bollinger Squeeze (BB Width rendah) → harga cenderung mau breakout.
    Setelah squeeze, pergerakan biasanya lebih kuat.

    Logika:
    - Harga di bawah lower band (oversold BB) + volume spike: skor tinggi
    - BB Squeeze terdeteksi (width di bawah percentile rendah): bonus
    - Harga di bawah lower band: moderate
    - Harga di antara bands: skor kecil
    - Harga di atas upper band (overbought BB): skor 0
    """

    name = "Bollinger Band"
    max_score = 10

    def compute(self, df: pd.DataFrame) -> dict:
        needed = {'bb_upper', 'bb_lower', 'bb_width', 'Close'}
        if not needed.issubset(df.columns):
            return self._no_data()

        valid = df.dropna(subset=['bb_upper', 'bb_lower', 'bb_width'])
        if valid.empty:
            return self._no_data()

        row = valid.iloc[-1]
        close    = row['Close']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        bb_width = row['bb_width']

        # Deteksi squeeze: width saat ini di bawah percentile 20 dari 50 hari terakhir
        width_history = valid['bb_width'].tail(50)
        squeeze = bb_width < width_history.quantile(0.2) if len(width_history) >= 10 else False
        squeeze_note = " + BB Squeeze terdeteksi (potensi breakout)" if squeeze else ""

        display = f"Close={close:.0f}, BBU={bb_upper:.0f}, BBL={bb_lower:.0f}, Width={bb_width:.4f}"

        # Harga di bawah lower band
        if close < bb_lower:
            return self._safe_result(
                display,
                f"Harga di bawah BB Lower — oversold zone secara statistik{squeeze_note}",
                self.max_score if squeeze else int(self.max_score * 0.8),
            )

        # Squeeze tanpa harga di luar band
        if squeeze:
            return self._safe_result(
                display,
                f"BB Squeeze — volatilitas rendah, potensi breakout explosive",
                int(self.max_score * 0.6),
            )

        # Harga di antara bands — cek posisi relatif
        band_range = bb_upper - bb_lower
        if band_range > 0:
            pos = (close - bb_lower) / band_range  # 0 = di lower, 1 = di upper
            if pos < 0.3:
                return self._safe_result(display, f"Harga dekat lower band (posisi {pos:.0%})", int(self.max_score * 0.4))
            if pos < 0.5:
                return self._safe_result(display, f"Harga di bawah tengah band (posisi {pos:.0%})", int(self.max_score * 0.2))

        # Harga di atas upper band (overbought)
        if close > bb_upper:
            return self._safe_result(display, f"Harga di atas BB Upper — overbought secara statistik", 0)

        return self._safe_result(display, "Posisi harga netral di dalam bands", int(self.max_score * 0.1))


# ─────────────────────────────────────────────────────────────────────────────
# Support/Resistance Scorer
# ─────────────────────────────────────────────────────────────────────────────

class SupportResistanceScorer(Indicator):
    """
    Skor berbasis posisi harga terhadap support/resistance level.
    S/R dihitung dari swing high-low N hari terakhir.

    Logika:
    - Harga breakout di atas resistance (konfirmasi): skor penuh
    - Harga mendekati resistance dari bawah (<2% away): skor parsial
    - Harga dekat support (bouncing): skor moderate
    - Harga di antara S/R tanpa sinyal: skor kecil
    """

    name = "Support/Resistance"
    max_score = 15

    def compute(self, df: pd.DataFrame) -> dict:
        needed = {'High', 'Low', 'Close'}
        if not needed.issubset(df.columns) or len(df) < 20:
            return self._no_data("Perlu minimal 20 hari data untuk hitung S/R")

        close = df['Close'].iloc[-1]
        sr_short = _cfg(None, 'sr_lookback_short')  # 20 hari
        sr_long  = _cfg(None, 'sr_lookback_long')   # 50 hari

        # Swing high-low dari N hari terakhir (exclude hari ini)
        history_short = df.iloc[-(sr_short + 1):-1]
        history_long  = df.iloc[-(sr_long + 1):-1] if len(df) > sr_long else df.iloc[:-1]

        resistance_20 = history_short['High'].max()
        support_20    = history_short['Low'].min()
        resistance_50 = history_long['High'].max()
        support_50    = history_long['Low'].min()

        display = f"Close={close:.0f} | R20={resistance_20:.0f}, S20={support_20:.0f}"

        # ── Breakout ──────────────────────────────────────────────────────────
        # Harga sudah di atas resistance 20 hari (breakout)
        if close > resistance_20:
            # Juga breakout dari resistance 50 hari? Sinyal lebih kuat.
            if close > resistance_50:
                return self._safe_result(
                    display,
                    f"Breakout di atas R50 ({resistance_50:.0f}) — sinyal breakout kuat",
                    self.max_score,
                )
            return self._safe_result(
                display,
                f"Breakout di atas R20 ({resistance_20:.0f}) — konfirmasi resistance break",
                int(self.max_score * 0.8),
            )

        # ── Mendekati Resistance ──────────────────────────────────────────────
        dist_to_r20 = (resistance_20 - close) / close
        if dist_to_r20 <= 0.02:  # Dalam 2% dari resistance
            return self._safe_result(
                display,
                f"Mendekati R20 ({resistance_20:.0f}, {dist_to_r20:.1%} lagi) — potensi breakout",
                int(self.max_score * 0.4),
            )

        # ── Dekat Support (Bouncing) ──────────────────────────────────────────
        dist_to_s20 = (close - support_20) / close
        prev_close  = df['Close'].iloc[-2] if len(df) >= 2 else close
        bouncing    = close > prev_close and dist_to_s20 <= 0.03  # Dalam 3% dari support, harga naik

        if bouncing:
            return self._safe_result(
                display,
                f"Memantul dari support S20 ({support_20:.0f}) — potential reversal",
                int(self.max_score * 0.6),
            )

        if dist_to_s20 <= 0.05:
            return self._safe_result(
                display,
                f"Harga dekat support S20 ({support_20:.0f}, {dist_to_s20:.1%} di atasnya)",
                int(self.max_score * 0.3),
            )

        # ── Tengah-tengah (no signal) ─────────────────────────────────────────
        return self._safe_result(
            display,
            f"Harga di antara S/R — tidak ada level kritis terdekat",
            int(self.max_score * 0.1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick Pattern Scorer
# ─────────────────────────────────────────────────────────────────────────────

class CandlestickScorer(Indicator):
    """
    Skor berbasis deteksi pola candlestick bullish (2–3 candle terakhir).

    Pola yang dideteksi:
    - Bullish Engulfing: candle hijau besar menelan candle merah sebelumnya
    - Hammer: lower shadow panjang (2x body) di zona oversold
    - Doji: body sangat kecil (ketidakpastian, neutral/reversal)

    Max skor kecil (5) karena candlestick hanya konfirmasi, bukan sinyal utama.
    """

    name = "Candlestick"
    max_score = 5

    def compute(self, df: pd.DataFrame) -> dict:
        needed = {'Open', 'High', 'Low', 'Close'}
        if not needed.issubset(df.columns) or len(df) < 2:
            return self._no_data("Perlu minimal 2 candle")

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        c_open  = curr['Open'];  c_high = curr['High']
        c_low   = curr['Low'];   c_close = curr['Close']
        p_open  = prev['Open'];  p_close = prev['Close']

        c_body  = abs(c_close - c_open)
        c_range = c_high - c_low if (c_high - c_low) > 0 else 1
        c_upper_shadow = c_high - max(c_open, c_close)
        c_lower_shadow = min(c_open, c_close) - c_low

        # ── Bullish Engulfing ────────────────────────────────────────────────
        # Candle saat ini: bullish (close > open)
        # Candle sebelumnya: bearish (close < open)
        # Body sekarang > body sebelumnya (engulf)
        if (c_close > c_open and
            p_close < p_open and
            c_open < p_close and
            c_close > p_open):
            return self._safe_result(
                f"Open={c_open:.0f}, Close={c_close:.0f}",
                "Bullish Engulfing — candle bullish menelan candle bearish sebelumnya",
                self.max_score,
            )

        # ── Hammer ──────────────────────────────────────────────────────────
        # Lower shadow ≥ 2× body, upper shadow kecil, bullish candle
        if (c_body > 0 and
            c_lower_shadow >= 2 * c_body and
            c_upper_shadow <= 0.3 * c_body and
            c_close >= c_open):
            return self._safe_result(
                f"Lower shadow={c_lower_shadow:.0f}, Body={c_body:.0f}",
                "Hammer — ekor panjang bawah, potensi reversal bullish",
                int(self.max_score * 0.8),
            )

        # ── Doji ─────────────────────────────────────────────────────────────
        # Body sangat kecil (< 10% dari range total)
        if c_range > 0 and c_body / c_range < 0.1:
            return self._safe_result(
                f"Body={c_body:.0f} ({c_body/c_range:.0%} of range)",
                "Doji — ketidakpastian, perhatikan candle berikutnya",
                int(self.max_score * 0.3),
            )

        # Tidak ada pola spesifik
        trend = "Bullish" if c_close >= c_open else "Bearish"
        return self._safe_result(
            f"{trend} candle",
            f"Tidak ada pola candlestick signifikan",
            0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_SCORERS: list[Indicator] = [
    EMACrossScorer(),
    VolumeScorer(),
    RSIScorer(),
    MACDScorer(),
    SupportResistanceScorer(),
    BollingerScorer(),
    CandlestickScorer(),
]
"""
Default list semua scorer yang dipakai ScoringEngine.
Total max score = sum of all max_score = 20+20+15+15+15+10+5 = 100.
"""
