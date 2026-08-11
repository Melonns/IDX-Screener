"""
rarity_context.py — Historical Rarity & Persistence Tracking (Descriptive, Non-Predictive)

Fungsi Utama:
1. `get_rarity_context()` : Menghitung berapa kali kondisi serupa terjadi pada saham yang SAMA
                           dalam 252 hari trading (12 bulan) terakhir, serta tanggal terakhir kali terjadi.
                           Menyertakan kategori frekuensi murni (relatif jarang vs relatif sering berulang).
2. `get_condition_streak()` : Menghitung berapa hari BERTURUT-TURUT kondisi tersebut sudah berlangsung.

Keduanya adalah FAKTA HISTORIS DESKRIPTIF murni, BUKAN prediksi probabilitas berulang.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


def get_rarity_context(
    ticker: str,
    condition_type: str,
    df_history: pd.DataFrame,
    lookback_days: int = 252
) -> Dict[str, Any]:
    """
    Hitung frekuensi kejadian serupa pada saham yang SAMA dalam 252 hari (12 bulan) terakhir.

    Args:
        ticker: Kode saham (misal 'BBRI.JK')
        condition_type: Kunci kondisi ('VOLUME_SPIKE', 'HIGH_RSI_RELATIVE', 'LOW_RSI_RELATIVE', 'BB_SQUEEZE', 'RESISTANCE_BREAKOUT')
        df_history: DataFrame harga & indikator historis (minimal 252 hari)
        lookback_days: Window histori dalam hari kerja (default: 252 hari ≈ 12 bulan)

    Returns:
        Dict: {'occurrences_count': int, 'last_occurrence_date': str or None, 'summary_text': str}
    """
    if len(df_history) < 20:
        return {'occurrences_count': 0, 'last_occurrence_date': None, 'summary_text': ""}

    df_sub = df_history.tail(lookback_days).copy()
    occurrences = []

    for i in range(len(df_sub)):
        sub_window = df_sub.iloc[:i+1]
        if len(sub_window) < 60:
            continue
        
        latest = sub_window.iloc[-1]
        dt_str = sub_window.index[-1].strftime('%Y-%m-%d')
        window_60 = sub_window.tail(60)

        is_match = False

        if condition_type == 'VOLUME_SPIKE':
            if 'volume_ratio_20d' in window_60.columns and pd.notna(latest['volume_ratio_20d']):
                rvol_series = window_60['volume_ratio_20d'].dropna()
                if len(rvol_series) > 10 and (rvol_series < latest['volume_ratio_20d']).mean() >= 0.90:
                    is_match = True

        elif condition_type == 'HIGH_RSI_RELATIVE':
            if 'rsi_14' in window_60.columns and pd.notna(latest['rsi_14']):
                rsi_series = window_60['rsi_14'].dropna()
                if len(rsi_series) > 10 and (rsi_series < latest['rsi_14']).mean() >= 0.90:
                    is_match = True

        elif condition_type == 'LOW_RSI_RELATIVE':
            if 'rsi_14' in window_60.columns and pd.notna(latest['rsi_14']):
                rsi_series = window_60['rsi_14'].dropna()
                if len(rsi_series) > 10 and (rsi_series < latest['rsi_14']).mean() <= 0.10:
                    is_match = True

        elif condition_type == 'BB_SQUEEZE':
            if 'bb_width' in window_60.columns and pd.notna(latest['bb_width']):
                bbw_series = window_60['bb_width'].dropna()
                if len(bbw_series) > 10 and (bbw_series < latest['bb_width']).mean() <= 0.10:
                    is_match = True

        elif condition_type == 'RESISTANCE_BREAKOUT':
            if len(sub_window) >= 21 and 'High' in sub_window.columns:
                high_20 = sub_window['High'].iloc[-21:-1].max()
                if latest['Close'] > high_20:
                    is_match = True

        if is_match:
            occurrences.append(dt_str)

    count = len(occurrences)
    # Exclude today's occurrence to find previous occurrence date
    prev_occurrences = occurrences[:-1] if count > 0 else []
    last_date = prev_occurrences[-1] if prev_occurrences else None

    # Neutral frequency descriptor
    freq_desc = "relatif jarang" if count <= 5 else "relatif sering berulang"

    if count > 0:
        if last_date:
            summary = f"Kejadian serupa: {count}x dalam 12 bulan ({freq_desc}, terakhir: {last_date})"
        else:
            summary = f"Kejadian serupa: {count}x dalam 12 bulan ({freq_desc})"
    else:
        summary = "Kejadian pertama dalam 12 bulan terakhir"

    return {
        'occurrences_count': count,
        'last_occurrence_date': last_date,
        'summary_text': summary
    }


def get_condition_streak(
    df_history: pd.DataFrame,
    condition_type: str
) -> int:
    """
    Hitung berapa hari BERTURUT-TURUT (streak) kondisi ini sudah berlangsung hingga hari ini.

    Returns:
        int: Jumlah hari beruntun (misal 1, 2, 4, dst.)
    """
    if len(df_history) < 60:
        return 1

    streak = 0
    for i in range(len(df_history) - 1, -1, -1):
        sub_window = df_history.iloc[:i+1]
        if len(sub_window) < 60:
            break
        
        latest = sub_window.iloc[-1]
        window_60 = sub_window.tail(60)
        is_match = False

        if condition_type == 'TREND_ALIGNMENT_UP':
            e9, e21, e50 = latest.get('ema_9'), latest.get('ema_21'), latest.get('ema_50')
            if pd.notna(e9) and pd.notna(e21) and pd.notna(e50) and e9 > e21 > e50:
                is_match = True

        elif condition_type == 'TREND_ALIGNMENT_DOWN':
            e9, e21, e50 = latest.get('ema_9'), latest.get('ema_21'), latest.get('ema_50')
            if pd.notna(e9) and pd.notna(e21) and pd.notna(e50) and e9 < e21 < e50:
                is_match = True

        elif condition_type == 'VOLUME_SPIKE':
            if 'volume_ratio_20d' in window_60.columns and pd.notna(latest['volume_ratio_20d']):
                rvol_s = window_60['volume_ratio_20d'].dropna()
                if len(rvol_s) > 10 and (rvol_s < latest['volume_ratio_20d']).mean() >= 0.90:
                    is_match = True

        elif condition_type == 'BB_SQUEEZE':
            if 'bb_width' in window_60.columns and pd.notna(latest['bb_width']):
                bbw_s = window_60['bb_width'].dropna()
                if len(bbw_s) > 10 and (bbw_s < latest['bb_width']).mean() <= 0.10:
                    is_match = True

        if is_match:
            streak += 1
        else:
            break

    return max(1, streak)
