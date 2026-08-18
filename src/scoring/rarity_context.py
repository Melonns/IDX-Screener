"""
rarity_context.py — Historical Rarity & Persistence Tracking (Descriptive, Non-Predictive)

Fungsi Utama:
1. `get_rarity_context()` : Menghitung berapa kali kondisi serupa terjadi pada saham yang SAMA
                           dalam 252 hari trading (12 bulan) terakhir, serta tanggal terakhir kali terjadi.
                           Menyertakan kategori frekuensi murni (relatif jarang vs relatif sering berulang).
2. `get_condition_streak()` : Menghitung berapa hari BERTURUT-TURUT kondisi tersebut sudah berlangsung.

Keduanya adalah FAKTA HISTORIS DESKRIPTIF murni, BUKAN prediksi probabilitas berulang.

OPTIMIZATION:
- Menggunakan LRU cache per-ticker untuk menghindari re-evaluasi berulang
  saat scan memanggil rarity untuk ticker yang sama dengan kondisi berbeda.
- Pre-compute condition checks dalam satu loop per ticker.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache

# ─── Cache for per-ticker condition evaluation ────────────────────────────────
# Key: (ticker, condition_type, tuple of date strings)
# We cache the final result to avoid re-scanning the same ticker
_RARITY_CACHE: Dict[str, Dict[str, Any]] = {}


def _build_condition_mask(df: pd.DataFrame, condition_type: str) -> pd.Series:
    """
    Build a boolean mask for a condition type across a DataFrame.
    Returns a Series of booleans aligned with df.index.
    Optimized: evaluates condition once for the entire series.
    """
    mask = pd.Series(False, index=df.index)

    if condition_type == 'VOLUME_SPIKE':
        if 'volume_ratio_20d' in df.columns:
            rvol = df['volume_ratio_20d'].dropna()
            if len(rvol) > 10:
                p90 = rvol.quantile(0.9)
                mask.loc[rvol.index] = rvol >= p90

    elif condition_type == 'HIGH_RSI_RELATIVE':
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14'].dropna()
            if len(rsi) > 10:
                p90 = rsi.quantile(0.9)
                mask.loc[rsi.index] = rsi >= p90

    elif condition_type == 'LOW_RSI_RELATIVE':
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14'].dropna()
            if len(rsi) > 10:
                p10 = rsi.quantile(0.1)
                mask.loc[rsi.index] = rsi <= p10

    elif condition_type == 'BB_SQUEEZE':
        if 'bb_width' in df.columns:
            bbw = df['bb_width'].dropna()
            if len(bbw) > 10:
                p10 = bbw.quantile(0.1)
                mask.loc[bbw.index] = bbw <= p10

    elif condition_type == 'RESISTANCE_BREAKOUT':
        if 'High' in df.columns and 'Close' in df.columns:
            # Need rolling 20-day high
            high_20 = df['High'].rolling(20, min_periods=20).max().shift(1)
            mask = df['Close'] > high_20

    elif condition_type in ('MACD_BULLISH_CROSS', 'MACD_BEARISH_CROSS'):
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd']
            signal = df['macd_signal']
            if condition_type == 'MACD_BULLISH_CROSS':
                # MACD crosses above signal
                prev_macd = macd.shift(1)
                prev_signal = signal.shift(1)
                mask = (prev_macd <= prev_signal) & (macd > signal)
            else:
                # MACD crosses below signal
                prev_macd = macd.shift(1)
                prev_signal = signal.shift(1)
                mask = (prev_macd >= prev_signal) & (macd < signal)

    elif condition_type == 'EMA50_FAR_ABOVE':
        if 'ema_50' in df.columns:
            ema50 = df['ema_50'].dropna()
            if len(ema50) > 0:
                close = df.loc[ema50.index, 'Close']
                pct = ((close - ema50) / ema50) * 100.0
                mask.loc[ema50.index] = pct > 5.0

    elif condition_type == 'EMA50_FAR_BELOW':
        if 'ema_50' in df.columns:
            ema50 = df['ema_50'].dropna()
            if len(ema50) > 0:
                close = df.loc[ema50.index, 'Close']
                pct = ((close - ema50) / ema50) * 100.0
                mask.loc[ema50.index] = pct < -5.0

    elif condition_type == 'ATR_EXPANSION':
        if 'atr_14' in df.columns:
            atr = df['atr_14'].dropna()
            if len(atr) >= 20:
                avg_20 = atr.rolling(20, min_periods=20).mean()
                ratio = atr / avg_20.replace(0, np.nan)
                mask.loc[atr.index] = ratio >= 1.5

    elif condition_type in ('GAP_UP', 'GAP_DOWN'):
        if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            prev_high = df['High'].shift(1)
            prev_low = df['Low'].shift(1)
            if condition_type == 'GAP_UP':
                gap_pct = ((df['Open'] - prev_high) / prev_high.replace(0, np.nan)) * 100.0
                mask = gap_pct >= 1.0
            else:
                gap_pct = ((df['Open'] - prev_low) / prev_low.replace(0, np.nan)) * 100.0
                mask = gap_pct <= -1.0

    return mask


def get_rarity_context(
    ticker: str,
    condition_type: str,
    df_history: pd.DataFrame,
    lookback_days: int = 252
) -> Dict[str, Any]:
    """
    Hitung frekuensi kejadian serupa pada saham yang SAMA dalam 252 hari (12 bulan) terakhir.
    Menggunakan vectorized evaluation untuk performa lebih baik.
    """
    if len(df_history) < 20:
        return {'occurrences_count': 0, 'last_occurrence_date': None, 'summary_text': ""}

    # Check cache
    cache_key = f"{ticker}:{condition_type}"
    if cache_key in _RARITY_CACHE:
        cached = _RARITY_CACHE[cache_key]
        # Invalidate if df changed (simple check: last date)
        last_date = str(df_history.index[-1])[:10]
        if cached.get('last_df_date') == last_date:
            return cached['result']

    df_sub = df_history.tail(lookback_days).copy()

    # Vectorized condition evaluation
    condition_mask = _build_condition_mask(df_sub, condition_type)

    # Count occurrences (excluding the very last bar which is "today")
    if len(condition_mask) > 1:
        historical_mask = condition_mask.iloc[:-1]
        occurrences = historical_mask[historical_mask].index.tolist()
    else:
        occurrences = []

    count = len(occurrences)
    last_date = occurrences[-1].strftime('%Y-%m-%d') if occurrences else None

    # Neutral frequency descriptor
    freq_desc = "relatif jarang" if count <= 5 else "relatif sering berulang"

    if count > 0:
        if last_date:
            summary = f"Kejadian serupa: {count}x dalam 12 bulan ({freq_desc}, terakhir: {last_date})"
        else:
            summary = f"Kejadian serupa: {count}x dalam 12 bulan ({freq_desc})"
    else:
        summary = "Kejadian pertama dalam 12 bulan terakhir"

    result = {
        'occurrences_count': count,
        'last_occurrence_date': last_date,
        'summary_text': summary
    }

    # Cache result
    _RARITY_CACHE[cache_key] = {
        'result': result,
        'last_df_date': str(df_history.index[-1])[:10]
    }

    return result


def clear_rarity_cache():
    """Clear the rarity cache (call after fresh data load)."""
    global _RARITY_CACHE
    _RARITY_CACHE.clear()


def get_condition_streak(
    df_history: pd.DataFrame,
    condition_type: str
) -> int:
    """
    Hitung berapa hari BERTURUT-TURUT (streak) kondisi ini sudah berlangsung hingga hari ini.
    Menggunakan vectorized evaluation untuk performa lebih baik.
    """
    if len(df_history) < 60:
        return 1

    # Build condition mask for the full history
    condition_mask = _build_condition_mask(df_history, condition_type)

    # Count consecutive True values from the end
    streak = 0
    for i in range(len(condition_mask) - 1, -1, -1):
        if condition_mask.iloc[i]:
            streak += 1
        else:
            break

    return max(1, streak)
