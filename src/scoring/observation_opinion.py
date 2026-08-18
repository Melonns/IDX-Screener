"""
observation_opinion.py — Contextual Observation Opinions (Non-Predictive)

Prinsip Utama:
- OPINI OBSERVASI, bukan sinyal beli/jual atau prediksi harga.
- Berdasarkan KOMBINASI kondisi teknikal yang terdeteksi pada hari ini.
- Setiap opini diawali dengan "Kondisi teknikal hari ini menunjukkan..." atau frasa serupa.
- Selalu dilengkapi kualifikasi risiko dan ajakan riset manual.
- TIDAK pernah menggunakan kata "BUY", "SELL", "HOLD", "TARGET PRICE", atau "PREDIKSI".

Kombinasi Tags → Opini:
  Kombinasi tag yang berbeda menghasilkan opini yang berbeda pula.
  Opini dibangun dari layer-layer:
    Layer 1: Momentum Direction (dari trend alignment)
    Layer 2: Volatility Context (dari BB squeeze, ATR expansion)
    Layer 3: Volume Confirmation (dari volume spike)
    Layer 4: Extreme Warning (dari RSI extreme, EMA50 distance)
    Layer 5: Fresh Signals (dari MACD cross, breakout/breakdown, gap)
"""

from typing import Dict, List, Any, Optional
import pandas as pd


def generate_observation_opinion(
    tags: List[Dict[str, Any]],
    close_price: float = 0,
    ticker: str = '',
) -> Dict[str, Any]:
    """
    Generate contextual observation opinion based on the combination of tags detected.

    Returns:
        Dict with keys:
            - headline: Short headline summary (1 line)
            - bullets: List of bullet point observations
            - overall_note: Overall contextual note
            - attention_level: 'TINGGI', 'SEDANG', or 'RENDAH'
    """
    tag_ids = {t['tag_id'] for t in tags}
    unusual_ids = {t['tag_id'] for t in tags if t.get('is_unusual')}

    if not unusual_ids:
        return {
            'headline': f"📊 {ticker}: Kondisi teknikal normal hari ini",
            'bullets': ["Tidak ada aktivitas di luar kebiasaan histori 60 hari."],
            'overall_note': "Saham dalam kondisi biasa. Tidak ada yang perlu diperhatikan khusus.",
            'attention_level': 'RENDAH',
        }

    bullets = []
    attention_score = 0

    # ── Layer 1: Momentum Direction ──────────────────────────────────────────
    if 'TREND_ALIGNMENT_UP' in tag_ids:
        bullets.append("📐 Struktur tren selaras naik (EMA9 > EMA21 > EMA50) — momentum sedang berpihak ke atas.")
        attention_score += 1
    elif 'TREND_ALIGNMENT_DOWN' in tag_ids:
        bullets.append("📐 Struktur tren selaras turun (EMA9 < EMA21 < EMA50) — momentum sedang berpihak ke bawah.")
        attention_score += 1

    # ── Layer 2: Volatility Context ──────────────────────────────────────────
    if 'BB_SQUEEZE' in unusual_ids:
        bullets.append("🔲 Bollinger Band menyempit — volatilitas rendah, potensi pergerakan besar (breakout/breakdown) dalam waktu dekat.")
        attention_score += 2

    if 'ATR_EXPANSION' in unusual_ids:
        bullets.append("📈 ATR meningkat signifikan — volatilitas sedang naik, antisipasi pergerakan harga yang lebih besar dari biasanya.")
        attention_score += 2

    # ── Layer 3: Volume Confirmation ─────────────────────────────────────────
    if 'VOLUME_SPIKE' in unusual_ids:
        bullets.append("📊 Volume perdagangan jauh di atas rata-rata — minat pasar terhadap saham ini meningkat hari ini.")
        attention_score += 2

        # Volume + Breakout = Stronger signal
        if 'RESISTANCE_BREAKOUT' in unusual_ids:
            bullets.append("⚡ Volume tinggi mengkonfirmasi breakout resistance — kondisi yang perlu diperhatikan serius.")
            attention_score += 1
        elif 'SUPPORT_BREAKDOWN' in unusual_ids:
            bullets.append("⚠️ Volume tinggi mengkonfirmasi breakdown support — berhati-hati dengan tekanan jual.")
            attention_score += 1

    # ── Layer 4: Extreme Warning ─────────────────────────────────────────────
    if 'HIGH_RSI_RELATIVE' in unusual_ids:
        rsi_val = next((t['description'] for t in tags if t['tag_id'] == 'HIGH_RSI_RELATIVE'), '')
        bullets.append(f"🔴 RSI sudah sangat tinggi relatif terhadap histori 60 hari — pergerakan cepat ke atas, namun potensi overbought.")
        attention_score += 1

    if 'LOW_RSI_RELATIVE' in unusual_ids:
        bullets.append("🟢 RSI sudah sangat rendah relatif terhadap histori 60 hari — pergerakan cepat ke bawah, potensi oversold.")
        attention_score += 1

    if 'EMA50_FAR_ABOVE' in unusual_ids:
        bullets.append("📏 Harga sudah cukup jauh di atas EMA50 — pertimbangkan bahwa harga sering kembali ke tren jangka panjang (mean reversion).")
        attention_score += 1

    if 'EMA50_FAR_BELOW' in unusual_ids:
        bullets.append("📏 Harga sudah cukup jauh di bawah EMA50 — pertimbangkan bahwa harga sering kembali ke tren jangka panjang (mean reversion).")
        attention_score += 1

    # ── Layer 5: Fresh Signals ───────────────────────────────────────────────
    if 'MACD_BULLISH_CROSS' in unusual_ids:
        bullets.append("🔄 MACD baru saja cross bullish — momentum baru terbentuk. Perlu dikonfirmasi dengan volume dan tren.")
        attention_score += 2
    elif 'MACD_BEARISH_CROSS' in unusual_ids:
        bullets.append("🔄 MACD baru saja cross bearish — momentum baru terbentuk ke bawah. Perlu dikonfirmasi.")
        attention_score += 2

    if 'RESISTANCE_BREAKOUT' in unusual_ids:
        bullets.append("🚀 Harga menembus tertinggi 20 hari — breakout yang perlu dipantau kelanjutannya.")
        attention_score += 2
    elif 'SUPPORT_BREAKDOWN' in unusual_ids:
        bullets.append("📉 Harga menembus terendah 20 hari — breakdown yang perlu dipantau kelanjutannya.")
        attention_score += 2

    if 'GAP_UP' in unusual_ids:
        bullets.append("⬆️ Gap up terjadi — market memberikan reaksi awal yang signifikan. Pantau apakah gap bertahan atau terisi.")
        attention_score += 1
    elif 'GAP_DOWN' in unusual_ids:
        bullets.append("⬇️ Gap down terjadi — market memberikan reaksi awal yang signifikan ke bawah. Pantau apakah gap bertahan.")
        attention_score += 1

    # ── Headline & Attention Level ───────────────────────────────────────────
    headline = _build_headline(tag_ids, unusual_ids, ticker)
    overall_note = _build_overall_note(tag_ids, unusual_ids, attention_score)

    if attention_score >= 5:
        attention_level = 'TINGGI'
    elif attention_score >= 3:
        attention_level = 'SEDANG'
    else:
        attention_level = 'RENDAH'

    return {
        'headline': headline,
        'bullets': bullets,
        'overall_note': overall_note,
        'attention_level': attention_level,
    }


def _build_headline(tag_ids: set, unusual_ids: set, ticker: str) -> str:
    """Build a concise headline based on the combination of tags."""
    # Priority-based headline selection
    if 'RESISTANCE_BREAKOUT' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        return f"⚡ {ticker}: Breakout Resistance + Volume Tinggi"
    elif 'SUPPORT_BREAKDOWN' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        return f"⚠️ {ticker}: Breakdown Support + Volume Tinggi"
    elif 'BB_SQUEEZE' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        return f"🔲 {ticker}: BB Squeeze + Volume Spike (Potensi Kejutan)"
    elif 'MACD_BULLISH_CROSS' in unusual_ids and 'TREND_ALIGNMENT_UP' in tag_ids:
        return f"🔄 {ticker}: MACD Cross Bullish + Tren Naik"
    elif 'MACD_BEARISH_CROSS' in unusual_ids and 'TREND_ALIGNMENT_DOWN' in tag_ids:
        return f"🔄 {ticker}: MACD Cross Bearish + Tren Turun"
    elif 'RESISTANCE_BREAKOUT' in unusual_ids:
        return f"🚀 {ticker}: Breakout Resistance"
    elif 'SUPPORT_BREAKDOWN' in unusual_ids:
        return f"📉 {ticker}: Breakdown Support"
    elif 'VOLUME_SPIKE' in unusual_ids:
        return f"📊 {ticker}: Volume Spike Signifikan"
    elif 'BB_SQUEEZE' in unusual_ids:
        return f"🔲 {ticker}: BB Squeeze (Volatilitas Rendah)"
    elif 'ATR_EXPANSION' in unusual_ids:
        return f"📈 {ticker}: Volatilitas Meningkat"
    elif 'MACD_BULLISH_CROSS' in unusual_ids:
        return f"🔄 {ticker}: MACD Cross Bullish"
    elif 'MACD_BEARISH_CROSS' in unusual_ids:
        return f"🔄 {ticker}: MACD Cross Bearish"
    elif 'HIGH_RSI_RELATIVE' in unusual_ids:
        return f"🔴 {ticker}: RSI Tinggi"
    elif 'LOW_RSI_RELATIVE' in unusual_ids:
        return f"🟢 {ticker}: RSI Rendah"
    elif 'EMA50_FAR_ABOVE' in unusual_ids:
        return f"📏 {ticker}: Harga Jauh di Atas EMA50"
    elif 'EMA50_FAR_BELOW' in unusual_ids:
        return f"📏 {ticker}: Harga Jauh di Bawah EMA50"
    elif 'GAP_UP' in unusual_ids:
        return f"⬆️ {ticker}: Gap Up"
    elif 'GAP_DOWN' in unusual_ids:
        return f"⬇️ {ticker}: Gap Down"
    else:
        return f"📊 {ticker}: Aktivitas di Luar Kebiasaan"


def _build_overall_note(tag_ids: set, unusual_ids: set, attention_score: int) -> str:
    """Build overall contextual note with appropriate qualifiers."""
    # Complex combinations get richer notes
    has_momentum = 'TREND_ALIGNMENT_UP' in tag_ids or 'TREND_ALIGNMENT_DOWN' in tag_ids
    has_volume = 'VOLUME_SPIKE' in unusual_ids
    has_volatility = 'BB_SQUEEZE' in unusual_ids or 'ATR_EXPANSION' in unusual_ids
    has_extreme = 'HIGH_RSI_RELATIVE' in unusual_ids or 'EMA50_FAR_ABOVE' in unusual_ids or 'EMA50_FAR_BELOW' in unusual_ids

    notes = []

    # Build context-aware overall note
    if has_momentum and has_volume:
        notes.append("Kombinasi tren dan volume menunjukkan partisipasi aktif dari pelaku pasar.")
    elif has_volume:
        notes.append("Peningkatan volume menunjukkan minat pasar yang meningkat.")

    if has_volatility:
        notes.append("Kondisi volatilitas menunjukkan potensi pergerakan besar dalam waktu dekat.")

    if has_extreme:
        notes.append("Kondisi ekstrelatif terhadap histori — perlu pertimbangan hati-hati.")

    if not notes:
        if attention_score >= 5:
            notes.append("Kombinasi beberapa kondisi unusual menunjukkan aktivitas yang perlu diperhatikan.")
        else:
            notes.append("Ada beberapa kondisi di luar kebiasaan yang bisa menjadi titik awal riset.")

    # Always add the disclaimer
    notes.append("💡 Ini adalah opini observasi teknikal, BUKAN rekomendasi beli/jual. Selalu lakukan riset fundamental dan konsultasi dengan penasihat keuangan sebelum keputusan investasi.")

    return " ".join(notes)


def format_opinion_telegram(opinion: Dict[str, Any]) -> str:
    """Format opinion for Telegram output."""
    lines = [
        f"💡 *Opini Observasi* — {opinion['headline']}",
        f"_Tingkat Perhatian: {opinion['attention_level']}_\n"
    ]

    for bullet in opinion['bullets']:
        lines.append(f"• {bullet}")

    lines.append(f"\n{opinion['overall_note']}")
    return "\n".join(lines)


def format_opinion_cli(opinion: Dict[str, Any]) -> str:
    """Format opinion for CLI output."""
    lines = [
        "="*70,
        f"  OPINI OBSERVASI — {opinion['headline']}",
        f"  Tingkat Perhatian: {opinion['attention_level']}",
        "="*70,
    ]

    for bullet in opinion['bullets']:
        lines.append(f"  • {bullet}")

    lines.append(f"\n  {opinion['overall_note']}")
    lines.append("="*70)
    return "\n".join(lines)
