"""
observation_tags.py — Descriptive Observation Tags & Liquidity Context (Non-Predictive)

Prinsip Utama:
- Deskriptif, bukan prediktif. Setiap tag adalah fakta terverifikasi, bukan sinyal/rekomendasi.
- Menggunakan ambang batas RELATIF terhadap histori 60 hari saham itu sendiri (persentil),
  bukan threshold absolut tunggal yang sama untuk semua saham.
- Menghindari kata "BULLISH", "BEARISH", "BUY", "SELL", atau skor 0-100.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def get_liquidity_note(turnover_5d: float | None) -> str:
    """
    Tampilkan angka turnover harian 5D sebagai fakta operasional eksekusi.
    Fakta tentang kondisi eksekusi, bukan penilaian kualitas saham.
    """
    if pd.isna(turnover_5d) or turnover_5d is None or turnover_5d <= 0:
        return "Turnover 5D: N/A"
    
    val_miliar = turnover_5d / 1_000_000_000.0
    if val_miliar >= 10.0:
        return f"Turnover 5D: Rp {val_miliar:.1f}M/hari (tergolong likuid)"
    else:
        return f"Turnover 5D: Rp {val_miliar:.1f}M/hari (likuiditas lebih rendah, slippage eksekusi mungkin lebih besar)"


def evaluate_observation_tags(df_stock: pd.DataFrame, lookback_days: int = 60) -> List[Dict[str, Any]]:
    """
    Evaluasi fakta teknikal harian pada baris terakhir `df_stock`
    berdasarkan distribusi relatif terhadap histori `lookback_days` saham itu sendiri.

    Returns:
        List of dicts containing tag details: {'tag_id': str, 'description': str, 'is_unusual': bool}
    """
    if len(df_stock) < lookback_days:
        return []

    # Get last lookback_days slice
    df_hist = df_stock.tail(lookback_days).copy()
    latest = df_hist.iloc[-1]

    tags = []

    # 1. VOLUME SPIKE (Volume Ratio 5D vs 60D history)
    if 'volume_ratio_20d' in df_hist.columns and pd.notna(latest['volume_ratio_20d']):
        rvol_series = df_hist['volume_ratio_20d'].dropna()
        if len(rvol_series) > 10:
            current_rvol = float(latest['volume_ratio_20d'])
            rvol_pct = (rvol_series < current_rvol).mean() * 100.0
            if rvol_pct >= 90.0:
                tags.append({
                    'tag_id': 'VOLUME_SPIKE',
                    'description': f"Volume: {current_rvol:.1f}x rata-rata (persentil {rvol_pct:.0f} dari 60 hari)",
                    'is_unusual': True
                })

    # 2. TREND ALIGNMENT (EMA9, EMA21, EMA50)
    ema9  = latest.get('ema_9')
    ema21 = latest.get('ema_21')
    ema50 = latest.get('ema_50')

    if pd.notna(ema9) and pd.notna(ema21) and pd.notna(ema50):
        if ema9 > ema21 > ema50:
            tags.append({
                'tag_id': 'TREND_ALIGNMENT_UP',
                'description': "Struktur Tren: EMA9 > EMA21 > EMA50 (selaras naik)",
                'is_unusual': False
            })
        elif ema9 < ema21 < ema50:
            tags.append({
                'tag_id': 'TREND_ALIGNMENT_DOWN',
                'description': "Struktur Tren: EMA9 < EMA21 < EMA50 (selaras turun)",
                'is_unusual': False
            })

    # 3. RESISTANCE & SUPPORT BREAKOUT / BREAKDOWN (20-day high/low)
    if len(df_hist) >= 20:
        high_20 = float(df_hist['High'].iloc[-21:-1].max()) if 'High' in df_hist.columns else None
        low_20  = float(df_hist['Low'].iloc[-21:-1].min())  if 'Low' in df_hist.columns  else None
        close_today = float(latest['Close'])

        if high_20 and close_today > high_20:
            tags.append({
                'tag_id': 'RESISTANCE_BREAKOUT',
                'description': f"Breakout: Menembus tertinggi 20 hari ({high_20:,.0f})",
                'is_unusual': True
            })
        elif low_20 and close_today < low_20:
            tags.append({
                'tag_id': 'SUPPORT_BREAKDOWN',
                'description': f"Breakdown: Menembus terendah 20 hari ({low_20:,.0f})",
                'is_unusual': True
            })

    # 4. RSI EXTREME RELATIVE TO STOCK'S OWN HISTORY
    if 'rsi_14' in df_hist.columns and pd.notna(latest['rsi_14']):
        rsi_series = df_hist['rsi_14'].dropna()
        if len(rsi_series) > 10:
            current_rsi = float(latest['rsi_14'])
            rsi_pct = (rsi_series < current_rsi).mean() * 100.0
            if rsi_pct >= 90.0:
                tags.append({
                    'tag_id': 'HIGH_RSI_RELATIVE',
                    'description': f"RSI: {current_rsi:.1f} (persentil {rsi_pct:.0f} dari 60 hari — pergerakan cepat)",
                    'is_unusual': True
                })
            elif rsi_pct <= 10.0:
                tags.append({
                    'tag_id': 'LOW_RSI_RELATIVE',
                    'description': f"RSI: {current_rsi:.1f} (persentil {rsi_pct:.0f} dari 60 hari — pergerakan cepat ke bawah)",
                    'is_unusual': True
                })

    # 5. BOLLINGER BAND SQUEEZE (Volatility contraction relative to own history)
    if 'bb_width' in df_hist.columns and pd.notna(latest['bb_width']):
        bbw_series = df_hist['bb_width'].dropna()
        if len(bbw_series) > 10:
            current_bbw = float(latest['bb_width'])
            bbw_pct = (bbw_series < current_bbw).mean() * 100.0
            if bbw_pct <= 10.0:
                tags.append({
                    'tag_id': 'BB_SQUEEZE',
                    'description': f"Bollinger Band: Menyempit (width persentil {bbw_pct:.0f} dari 60 hari — volatilitas rendah, arah tidak diketahui)",
                    'is_unusual': True
                })

    return tags
