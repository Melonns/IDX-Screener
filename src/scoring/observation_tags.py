"""
observation_tags.py — Descriptive Observation Tags & Liquidity Context (Non-Predictive)

Prinsip Utama:
- Deskriptif, bukan prediktif. Setiap tag adalah fakta terverifikasi, bukan sinyal/rekomendasi.
- Menggunakan ambang batas RELATIF terhadap histori 60 hari saham itu sendiri (persentil),
  bukan threshold absolut tunggal yang sama untuk semua saham.
- Menghindari kata "BULLISH", "BEARISH", "BUY", "SELL", atau skor 0-100.

Tags (10 total):
1. VOLUME_SPIKE        — Volume ratio >= persentil 90 dari 60 hari
2. TREND_ALIGNMENT_UP/DOWN — EMA9/21/50 alignment
3. RESISTANCE_BREAKOUT — Close > tertinggi 20 hari
4. SUPPORT_BREAKDOWN   — Close < terendah 20 hari
5. HIGH_RSI_RELATIVE   — RSI >= persentil 90 dari 60 hari
6. LOW_RSI_RELATIVE    — RSI <= persentil 10 dari 60 hari
7. BB_SQUEEZE          — BB width <= persentil 10 dari 60 hari
8. MACD_BULLISH_CROSS  — MACD cross above signal (fresh, <=3 hari)
9. MACD_BEARISH_CROSS  — MACD cross below signal (fresh, <=3 hari)
10. EMA50_FAR_ABOVE    — Close > 5% di atas EMA50
11. EMA50_FAR_BELOW    — Close < 5% di bawah EMA50
12. ATR_EXPANSION      — ATR naik >= 50% dari rata-rata 20 hari (volatilitas spike)
13. GAP_UP             — Open > High前一天 (gap up signifikan)
14. GAP_DOWN           — Open < Low前一天 (gap down signifikan)
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

    # ── 1. VOLUME SPIKE (Volume Ratio vs 60D history) ──────────────────────────
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

    # ── 2. TREND ALIGNMENT (EMA9, EMA21, EMA50) ───────────────────────────────
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

    # ── 3. RESISTANCE & SUPPORT BREAKOUT / BREAKDOWN (20-day high/low) ─────────
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

    # ── 4. RSI EXTREME RELATIVE TO STOCK'S OWN HISTORY ─────────────────────────
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

    # ── 5. BOLLINGER BAND SQUEEZE (Volatility contraction) ─────────────────────
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

    # ── 6. MACD CROSS (Fresh momentum shift) ──────────────────────────────────
    if 'macd' in df_hist.columns and 'macd_signal' in df_hist.columns:
        macd_val = latest.get('macd')
        signal_val = latest.get('macd_signal')
        if pd.notna(macd_val) and pd.notna(signal_val) and len(df_hist) >= 3:
            # Check for recent cross (within last 3 days)
            for lookback in range(1, 4):
                if len(df_hist) > lookback:
                    prev = df_hist.iloc[-(lookback + 1)]
                    prev_macd = prev.get('macd')
                    prev_signal = prev.get('macd_signal')
                    if pd.notna(prev_macd) and pd.notna(prev_signal):
                        # Bullish cross: MACD was below signal, now above
                        if prev_macd <= prev_signal and macd_val > signal_val:
                            tags.append({
                                'tag_id': 'MACD_BULLISH_CROSS',
                                'description': f"MACD: Cross bullish ({lookback} hari lalu, MACD {macd_val:.2f} > Signal {signal_val:.2f})",
                                'is_unusual': True
                            })
                            break
                        # Bearish cross: MACD was above signal, now below
                        elif prev_macd >= prev_signal and macd_val < signal_val:
                            tags.append({
                                'tag_id': 'MACD_BEARISH_CROSS',
                                'description': f"MACD: Cross bearish ({lookback} hari lalu, MACD {macd_val:.2f} < Signal {signal_val:.2f})",
                                'is_unusual': True
                            })
                            break

    # ── 7. EMA50 DISTANCE (Far above/below long-term trend) ────────────────────
    if pd.notna(ema50) and ema50 > 0:
        close_val = float(latest['Close'])
        pct_from_ema50 = ((close_val - ema50) / ema50) * 100.0

        if pct_from_ema50 > 5.0:
            tags.append({
                'tag_id': 'EMA50_FAR_ABOVE',
                'description': f"Harga: {pct_from_ema50:+.1f}% di atas EMA50 ({ema50:,.0f}) — sudah jauh dari tren jangka panjang",
                'is_unusual': True
            })
        elif pct_from_ema50 < -5.0:
            tags.append({
                'tag_id': 'EMA50_FAR_BELOW',
                'description': f"Harga: {pct_from_ema50:+.1f}% di bawah EMA50 ({ema50:,.0f}) — sudah jauh dari tren jangka panjang",
                'is_unusual': True
            })

    # ── 8. ATR EXPANSION (Volatility spike) ───────────────────────────────────
    if 'atr_14' in df_hist.columns and pd.notna(latest['atr_14']):
        atr_series = df_hist['atr_14'].dropna()
        if len(atr_series) >= 20:
            current_atr = float(latest['atr_14'])
            avg_atr_20 = float(atr_series.iloc[-20:].mean())
            if avg_atr_20 > 0:
                atr_ratio = current_atr / avg_atr_20
                if atr_ratio >= 1.5:
                    tags.append({
                        'tag_id': 'ATR_EXPANSION',
                        'description': f"ATR: {atr_ratio:.1f}x rata-rata 20 hari (volatilitas meningkat signifikan)",
                        'is_unusual': True
                    })

    # ── 9. GAP UP / GAP DOWN ──────────────────────────────────────────────────
    if len(df_hist) >= 2:
        prev_bar = df_hist.iloc[-2]
        today_open = float(latest['Open'])
        prev_high = float(prev_bar['High'])
        prev_low  = float(prev_bar['Low'])
        prev_close = float(prev_bar['Close'])

        # Gap Up: Open today > High yesterday (gap yang signifikan)
        if prev_high > 0:
            gap_up_pct = ((today_open - prev_high) / prev_high) * 100.0
            if gap_up_pct >= 1.0:
                tags.append({
                    'tag_id': 'GAP_UP',
                    'description': f"Gap Up: {gap_up_pct:+.1f}% dari penutupan kemarin",
                    'is_unusual': True
                })

        # Gap Down: Open today < Low yesterday
        if prev_low > 0:
            gap_down_pct = ((today_open - prev_low) / prev_low) * 100.0
            if gap_down_pct <= -1.0:
                tags.append({
                    'tag_id': 'GAP_DOWN',
                    'description': f"Gap Down: {gap_down_pct:+.1f}% dari terendah kemarin",
                    'is_unusual': True
                })

    return tags
