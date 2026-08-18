"""
observation_opinion.py — Contextual Observation Opinions (Non-Predictive, CONCISE)

Prinsip Utama:
- OPINI OBSERVASI, bukan sinyal beli/jual atau prediksi harga.
- Hanya menampilkan COMBO INSIGHTS — apa arti dari KOMBINASI tags yang terdeteksi.
- TIDAK mengulang deskripsi individual tags (itu sudah ada di bagian tags).
- Max 3 bullet points + 1 closing note. Singkat, to the point.
- TIDAK pernah menggunakan kata "BUY", "SELL", "HOLD", "TARGET PRICE", atau "PREDIKSI".
"""

from typing import Dict, List, Any


def generate_observation_opinion(
    tags: List[Dict[str, Any]],
    close_price: float = 0,
    ticker: str = '',
) -> Dict[str, Any]:
    """
    Generate concise observation opinion — combo insights only, no tag repetition.
    """
    tag_ids = {t['tag_id'] for t in tags}
    unusual_ids = {t['tag_id'] for t in tags if t.get('is_unusual')}

    if not unusual_ids:
        return {
            'headline': f"📊 {ticker}: Normal",
            'bullets': [],
            'closing': "Kondisi teknikal biasa. Tidak ada yang perlu diperhatikan khusus.",
            'attention_level': 'RENDAH',
        }

    bullets = []
    attention_score = 0

    # ── Combo Insights (bukan ulang tag individual) ──────────────────────────

    # Combo: Breakout + Volume
    if 'RESISTANCE_BREAKOUT' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        bullets.append("⚡ Breakout dikonfirmasi volume tinggi — kondisi yang perlu diperhatikan serius.")
        attention_score += 3
    elif 'SUPPORT_BREAKDOWN' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        bullets.append("⚠️ Breakdown dikonfirmasi volume tinggi — berhati-hati dengan tekanan jual.")
        attention_score += 3

    # Combo: BB Squeeze + Volume (potensi kejutan)
    elif 'BB_SQUEEZE' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        bullets.append("🔲 BB Squeeze + Volume Spike — potensi breakout/breakdown dalam waktu dekat.")
        attention_score += 3

    # Combo: MACD Cross + Trend alignment
    if 'MACD_BULLISH_CROSS' in unusual_ids and 'TREND_ALIGNMENT_UP' in tag_ids:
        bullets.append("🔄 MACD cross bullish dalam tren naik — momentum baru terkonfirmasi.")
        attention_score += 2
    elif 'MACD_BEARISH_CROSS' in unusual_ids and 'TREND_ALIGNMENT_DOWN' in tag_ids:
        bullets.append("🔄 MACD cross bearish dalam tren turun — momentum baru terkonfirmasi ke bawah.")
        attention_score += 2
    elif 'MACD_BULLISH_CROSS' in unusual_ids:
        bullets.append("🔄 MACD cross bullish — perlu konfirmasi dari tren dan volume.")
        attention_score += 2
    elif 'MACD_BEARISH_CROSS' in unusual_ids:
        bullets.append("🔄 MACD cross bearish — perlu konfirmasi dari tren dan volume.")
        attention_score += 2

    # Extreme warnings (hanya kalau belum dicover combo di atas)
    if 'HIGH_RSI_RELATIVE' in unusual_ids and 'EMA50_FAR_ABOVE' in unusual_ids:
        if attention_score < 4:
            bullets.append("🔴 RSI tinggi + harga jauh di atas EMA50 — potensi koreksi/retracement.")
            attention_score += 2
    elif 'HIGH_RSI_RELATIVE' in unusual_ids:
        if attention_score < 3:
            bullets.append("🔴 RSI sudah ekstrem tinggi — potensi overbought.")
            attention_score += 1
    elif 'LOW_RSI_RELATIVE' in unusual_ids:
        if attention_score < 3:
            bullets.append("🟢 RSI sudah ekstrem rendah — potensi oversold.")
            attention_score += 1

    if 'EMA50_FAR_ABOVE' in unusual_ids and attention_score < 3:
        bullets.append("📏 Harga sudah jauh di atas EMA50 — potensi mean reversion.")
        attention_score += 1
    elif 'EMA50_FAR_BELOW' in unusual_ids and attention_score < 3:
        bullets.append("📏 Harga sudah jauh di bawah EMA50 — potensi mean reversion.")
        attention_score += 1

    # ATR expansion standalone
    if 'ATR_EXPANSION' in unusual_ids and attention_score < 3:
        bullets.append("📈 Volatilitas meningkat signifikan — antisipasi pergerakan besar.")
        attention_score += 2

    # Gap (standalone, only if not part of bigger combo)
    if 'GAP_UP' in unusual_ids and attention_score < 2:
        bullets.append("⬆️ Gap up — pantau apakah bertahan atau terisi.")
        attention_score += 1
    elif 'GAP_DOWN' in unusual_ids and attention_score < 2:
        bullets.append("⬇️ Gap down — pantau apakah bertahan.")
        attention_score += 1

    # Limit bullets to max 3
    bullets = bullets[:3]

    # Closing note — singkat, tanpa disclaimer (udah di bawah report)
    if attention_score >= 5:
        closing = "Kombinasi kondisi yang signifikan — lakukan riset lebih lanjut sebelum keputusan."
    elif attention_score >= 3:
        closing = "Ada beberapa kondisi menarik yang layak dipantau."
    else:
        closing = "Kondisi perlu dipantau, belum mendesak."

    # Attention level
    if attention_score >= 5:
        attention_level = 'TINGGI'
    elif attention_score >= 3:
        attention_level = 'SEDANG'
    else:
        attention_level = 'RENDAH'

    headline = _build_headline(tag_ids, unusual_ids, ticker)

    return {
        'headline': headline,
        'bullets': bullets,
        'closing': closing,
        'attention_level': attention_level,
    }


def _build_headline(tag_ids: set, unusual_ids: set, ticker: str) -> str:
    """Build a concise headline."""
    if 'RESISTANCE_BREAKOUT' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        return f"⚡ {ticker}: Breakout + Volume"
    elif 'SUPPORT_BREAKDOWN' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        return f"⚠️ {ticker}: Breakdown + Volume"
    elif 'BB_SQUEEZE' in unusual_ids and 'VOLUME_SPIKE' in unusual_ids:
        return f"🔲 {ticker}: Squeeze + Volume"
    elif 'MACD_BULLISH_CROSS' in unusual_ids and 'TREND_ALIGNMENT_UP' in tag_ids:
        return f"🔄 {ticker}: MACD Bullish + Tren Naik"
    elif 'MACD_BEARISH_CROSS' in unusual_ids and 'TREND_ALIGNMENT_DOWN' in tag_ids:
        return f"🔄 {ticker}: MACD Bearish + Tren Turun"
    elif 'RESISTANCE_BREAKOUT' in unusual_ids:
        return f"🚀 {ticker}: Breakout"
    elif 'SUPPORT_BREAKDOWN' in unusual_ids:
        return f"📉 {ticker}: Breakdown"
    elif 'VOLUME_SPIKE' in unusual_ids:
        return f"📊 {ticker}: Volume Spike"
    elif 'MACD_BULLISH_CROSS' in unusual_ids:
        return f"🔄 {ticker}: MACD Bullish"
    elif 'MACD_BEARISH_CROSS' in unusual_ids:
        return f"🔄 {ticker}: MACD Bearish"
    elif 'HIGH_RSI_RELATIVE' in unusual_ids:
        return f"🔴 {ticker}: RSI Tinggi"
    elif 'LOW_RSI_RELATIVE' in unusual_ids:
        return f"🟢 {ticker}: RSI Rendah"
    elif 'BB_SQUEEZE' in unusual_ids:
        return f"🔲 {ticker}: BB Squeeze"
    elif 'ATR_EXPANSION' in unusual_ids:
        return f"📈 {ticker}: Volatilitas Naik"
    elif 'EMA50_FAR_ABOVE' in unusual_ids:
        return f"📏 {ticker}: Jauh di Atas EMA50"
    elif 'EMA50_FAR_BELOW' in unusual_ids:
        return f"📏 {ticker}: Jauh di Bawah EMA50"
    elif 'GAP_UP' in unusual_ids:
        return f"⬆️ {ticker}: Gap Up"
    elif 'GAP_DOWN' in unusual_ids:
        return f"⬇️ {ticker}: Gap Down"
    else:
        return f"📊 {ticker}: Aktivitas Unusual"


def format_opinion_telegram(opinion: Dict[str, Any]) -> str:
    """Format opinion for Telegram — concise."""
    lines = [f"💡 *{opinion['headline']}* [{opinion['attention_level']}]"]
    for b in opinion['bullets']:
        lines.append(f"• {b}")
    lines.append(f"_{opinion['closing']}_")
    return "\n".join(lines)


def format_opinion_cli(opinion: Dict[str, Any]) -> str:
    """Format opinion for CLI — concise."""
    lines = [
        f"  💡 {opinion['headline']} [{opinion['attention_level']}]",
    ]
    for b in opinion['bullets']:
        lines.append(f"     • {b}")
    lines.append(f"     {opinion['closing']}")
    return "\n".join(lines)
