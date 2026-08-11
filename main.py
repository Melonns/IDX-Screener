"""
main.py — Flask & Telegram Bot Server for IDX-Screener v3

Fitur Telegram Bot v3:
1. /scan atau pesan otomatis: Scan seluruh universe saham pasar BEI (800+ saham) & tampilkan 5-15 saham
   beraktivitas di luar kebiasaan (Unusual Activity) dengan fakta deskriptif + Rarity + Market Breadth + Sector Context.
   Dilengkapi pesan loading interaktif (editMessageText & typing indicator).
2. Input ticker (misal: BBRI ASII): Tampilkan fakta observasi deskriptif khusus saham tersebut.
3. /dividends: Tampilkan sinyal aktif strategi Dividend Drift v1.0 yang divalidasi out-of-sample.
4. /ping: Tes koneksi & latensi jaringan bot real-time.

CATATAN:
- TIDAK ADA label "BULLISH", "BEARISH", "BUY", "SELL", atau skor 0-100.
- Setiap output menyertakan Peringatan Wajib secara transparan.
"""

import io
import json
import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import requests
from flask import Flask, jsonify, request

_HERE = Path(__file__).parent
_SRC  = _HERE / 'src'
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from data.database import DatabaseManager
from scoring.scanner import TechnicalObservationScanner
from data.forward_tracker import DividendForwardTracker
from data.ingestion import compute_indicators
import config as app_config

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_API_URL   = 'https://api.telegram.org'

db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
db = DatabaseManager(db_path)
scanner = TechnicalObservationScanner(db)
div_tracker = DividendForwardTracker(db)


def send_telegram_message(chat_id: int, text: str) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not configured')
    url = f'{TELEGRAM_API_URL}/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}
    resp = requests.post(url, json=payload, timeout=15)
    return resp.json()


def edit_telegram_message(chat_id: int, message_id: int, text: str) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not configured')
    url = f'{TELEGRAM_API_URL}/bot{TELEGRAM_BOT_TOKEN}/editMessageText'
    payload = {'chat_id': chat_id, 'message_id': message_id, 'text': text, 'parse_mode': 'Markdown'}
    resp = requests.post(url, json=payload, timeout=15)
    return resp.json()


def send_chat_action(chat_id: int, action: str = 'typing') -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not configured')
    url = f'{TELEGRAM_API_URL}/bot{TELEGRAM_BOT_TOKEN}/sendChatAction'
    payload = {'chat_id': chat_id, 'action': action}
    resp = requests.post(url, json=payload, timeout=15)
    return resp.json()


def get_help_message() -> str:
    """Return updated help message for Telegram bot v3."""
    return """🤖 *IDX-Screener v3 — Daily Technical Observation & Dividend Tracker*

📌 *Cara Penggunaan Telegram Bot:*

1️⃣ */scan* (atau kirim tombol Scan):
   Memindai seluruh pasar BEI (800+ saham) & menampilkan daftar 5–15 saham yang beraktivitas di luar kebiasaan (dengan indicator loading interaktif).

2️⃣ *Kirim Kode Saham* (misal: `BBRI` atau `ASII TLKM`):
   Melihat fakta observasi deskriptif (Volume percentile, RSI percentile, EMA trend, Breakout, Rarity 12 bulan) khusus saham tersebut.

3️⃣ */dividends*:
   Melihat sinyal aktif & mendatang dari strategi *Dividend Cum-Date Drift* (strategi yang lolos validasi out-of-sample).

4️⃣ */ping*:
   Tes koneksi jaringan, status server, dan latensi bot real-time.

*Perintah Tambahan:*
• /start - Mulai bot
• /help - Petunjuk ini

⚠️ *Catatan:* Bot ini menyajikan fakta deskriptif sebagai alat observasi riset manual, BUKAN sinyal prediktif beli/jual otomatis.
"""


def parse_tickers_from_text(text: str) -> list[str]:
    text = text.replace(',', ' ')
    tokens = re.findall(r'[A-Za-z0-9\.]+', text.upper())
    tickers = []
    for token in tokens:
        token = token.strip()
        if token.endswith('JK') and not token.endswith('.JK'):
            token = token[:-2] + '.JK'
        if token.endswith('.JK') or (len(token) == 4 and token.isalpha()):
            tickers.append(token)
    return list(dict.fromkeys(tickers))


@app.route('/')
def status():
    return jsonify({
        'status': 'ok',
        'system': 'IDX-Screener v3 — Daily Observation & Dividend Tracker',
        'telegram_bot': bool(TELEGRAM_BOT_TOKEN),
    })


@app.route('/scan_api', methods=['GET'])
def scan_api():
    max_res = int(request.args.get('max_results', 10))
    results = scanner.scan_unusual_activity(max_results=max_res)
    return jsonify({
        'status': 'ok',
        'count': len(results),
        'results': results
    })


@app.route('/telegram_webhook', methods=['POST'])
def telegram_webhook():
    t_start = time.time()
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({'error': 'TELEGRAM_BOT_TOKEN is not configured'}), 500

    data = request.get_json(force=True)
    message = data.get('message') or data.get('edited_message')
    if not message:
        return jsonify({'ok': False, 'error': 'No message received'}), 200

    chat_id = message['chat']['id']
    text = message.get('text', '').strip()

    if not text:
        send_telegram_message(chat_id, get_help_message())
        return jsonify({'ok': True})

    text_lower = text.lower()

    # Handle /start or /help
    if text_lower in ['/start', '/help']:
        send_telegram_message(chat_id, get_help_message())
        return jsonify({'ok': True})

    # Handle /ping command (network & latency test)
    if text_lower in ['/ping', 'ping', 'tes', 'test']:
        send_chat_action(chat_id, 'typing')
        latency_ms = int((time.time() - t_start) * 1000)
        
        # Check SQLite DB connection
        try:
            db_tickers_count = len(db.get_tickers())
            db_status_str = f"Normal ({db_tickers_count} saham di-cache)"
        except Exception:
            db_status_str = "Koneksi DB Bermasalah"

        ping_msg = (
            f"🏓 *Pong! Jaringan Bot Normal*\n\n"
            f"• **Latensi Jaringan** : `{latency_ms} ms`\n"
            f"• **Status Server**    : `Online ✅`\n"
            f"• **Status Database**  : `{db_status_str}`\n"
            f"• **Versi Bot**        : `IDX-Screener v3 (Descriptive)`"
        )
        send_telegram_message(chat_id, ping_msg)
        return jsonify({'ok': True})

    # Handle /scan command with INTERACTIVE LOADING MESSAGE
    if text_lower in ['/scan', 'scan', 'scan hari ini']:
        send_chat_action(chat_id, 'typing')
        
        # 1. Send loading message
        load_res = send_telegram_message(
            chat_id,
            "⏳ *Sedang memindai seluruh saham pasar BEI (800+ saham)...*\n"
            "_Proses ini mengunduh & mengevaluasi fakta 60-hari. Harap tunggu beberapa detik._"
        )
        msg_id = load_res.get('result', {}).get('message_id')

        # 2. Run scan
        send_chat_action(chat_id, 'typing')
        results = scanner.scan_unusual_activity(max_results=10)
        report  = scanner.format_telegram_report(results)

        # 3. Edit loading message into final report if message_id exists, else send new
        if msg_id:
            try:
                edit_telegram_message(chat_id, msg_id, report)
            except Exception:
                send_telegram_message(chat_id, report)
        else:
            send_telegram_message(chat_id, report)

        return jsonify({'ok': True})

    # Handle /dividends command with INTERACTIVE LOADING MESSAGE
    if text_lower in ['/dividends', 'dividend', 'dividen']:
        send_chat_action(chat_id, 'typing')
        
        load_res = send_telegram_message(
            chat_id,
            "⏳ *Sedang memuat data sinyal Dividend Cum-Date Drift (V3_LOCKED)...*"
        )
        msg_id = load_res.get('result', {}).get('message_id')

        df_act = div_tracker.scan_upcoming_signals()
        if not df_act.empty:
            active_list = []
            for _, r in df_act.iterrows():
                active_list.append(
                    f"💰 *{r['ticker'].replace('.JK', '')}*\n"
                    f"   • Entry Date : `{r['entry_date']}`\n"
                    f"   • Cum-Date   : `{r['cum_date']}`\n"
                    f"   • Yield      : `{r['yield']:.2f}%`\n"
                    f"   • Turnover 5D: `Rp {r['turnover_5d']/1e9:.1f}M/hari`"
                )
            msg = "📊 *Sinyal Aktif Dividend Cum-Date Drift (V3_LOCKED)*\n\n" + "\n\n".join(active_list) + "\n\n⚠️ *Strategi ini lolos validasi out-of-sample holdout.*"
        else:
            msg = "💰 *Dividend Cum-Date Drift Tracker*\n\nSaat ini tidak ada event dividen qualified (Yield >= 4.0%) yang masuk periode entry."

        if msg_id:
            try:
                edit_telegram_message(chat_id, msg_id, msg)
            except Exception:
                send_telegram_message(chat_id, msg)
        else:
            send_telegram_message(chat_id, msg)

        return jsonify({'ok': True})

    # Handle specific Tickers input (e.g. ASII BBCA) with INTERACTIVE LOADING MESSAGE
    tickers = parse_tickers_from_text(text)
    if not tickers:
        send_telegram_message(chat_id, '❓ Perintah atau kode saham tidak dikenali.\n\nGunakan /scan, /ping, /dividends, atau ketik kode saham (misal: `BBRI ASII`).')
        return jsonify({'ok': True})

    send_chat_action(chat_id, 'typing')
    ticker_str = ", ".join([t.replace('.JK', '') for t in tickers])
    load_res = send_telegram_message(
        chat_id,
        f"⏳ *Sedang memproses & mengambil data observasi {ticker_str}...*"
    )
    msg_id = load_res.get('result', {}).get('message_id')

    results = scanner.scan_unusual_activity(tickers=tickers, max_results=len(tickers))
    
    if results:
        report = scanner.format_telegram_report(results)
    else:
        # If tickers have no unusual activity today, display basic status & auto-fetch if needed
        lines = [f"📋 **Fakta Observasi Teknikal ({tickers[0].replace('.JK','')})**\n"]
        for t in tickers:
            t_clean = t if t.endswith('.JK') else f"{t}.JK"
            df = db.get_prices_with_indicators(t_clean)
            if df.empty or len(df) < 60:
                try:
                    df_p = scanner.provider.get_or_fetch(t_clean, period_days=252*2)
                    if not df_p.empty:
                        df_ind = compute_indicators(df_p)
                        db.save_indicators(t_clean, df_ind)
                        df = db.get_prices_with_indicators(t_clean)
                except Exception:
                    pass

            if not df.empty:
                last_row = df.iloc[-1]
                close_p  = float(last_row['Close'])
                rsi_val  = float(last_row['rsi_14']) if 'rsi_14' in last_row and pd.notna(last_row['rsi_14']) else 'N/A'
                rsi_str  = f"{rsi_val:.1f}" if isinstance(rsi_val, float) else rsi_val
                lines.append(f"• **{t.replace('.JK', '')}** (Rp {close_p:,.0f}): RSI 14 = {rsi_str} — Tidak ada aktivitas di luar kebiasaan histori 60 hari hari ini.")
            else:
                lines.append(f"• **{t}**: Data tidak ditemukan (gagal fetch).")
        lines.append(f"\n---\n{scanner.MANDATORY_DISCLAIMER if hasattr(scanner, 'MANDATORY_DISCLAIMER') else ''}")
        report = "\n".join(lines)

    if msg_id:
        try:
            edit_telegram_message(chat_id, msg_id, report)
        except Exception:
            send_telegram_message(chat_id, report)
    else:
        send_telegram_message(chat_id, report)

    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
