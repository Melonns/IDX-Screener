"""
main.py — Flask & Telegram Bot Server for IDX-Screener v3

Fitur Telegram Bot v3:
1. /scan atau pesan otomatis: Scan seluruh universe saham pasar BEI dengan Async Background Worker (threading).
   Dilengkapi Safety Net (Anti-Spam Lock) & Automatic Plain Text Fallback jika Telegram Markdown error.
2. /status atau /progress: Cek status & persentase progres pemindaian real-time saat dipanggil user.
3. Input ticker (misal: BBRI ASII): Tampilkan fakta observasi deskriptif khusus saham tersebut (Async Worker).
4. /dividends: Tampilkan sinyal aktif strategi Dividend Drift v1.0 yang divalidasi out-of-sample.
5. /ping: Tes koneksi & latensi jaringan bot real-time.

CATATAN:
- TIDAK ADA label "BULLISH", "BEARISH", "BUY", "SELL", atau skor 0-100.
- Setiap output menyertakan Peringatan Wajib secara transparan.
"""

import io
import traceback
import json
import os
import re
import sys
import time
import threading
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

# Safety Net Lock — Mencegah spam scan ganda per chat_id
ACTIVE_SCANS = set()
ACTIVE_SCANS_LOCK = threading.Lock()


def send_telegram_message(chat_id: int, text: str) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not configured')
    url = f'{TELEGRAM_API_URL}/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}
    resp = requests.post(url, json=payload, timeout=15)
    res_json = resp.json()

    # Fallback to plain text if Telegram rejects Markdown formatting (e.g. unescaped underscores)
    if not res_json.get('ok'):
        payload.pop('parse_mode', None)
        resp = requests.post(url, json=payload, timeout=15)
        res_json = resp.json()

    return res_json


def escape_markdown_v1(text: str) -> str:
    """Escape special characters for Telegram MarkdownV1."""
    # Characters that need escaping in MarkdownV1: _ * ` [
    # But we use these intentionally, so only escape in dynamic content
    return text.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')


def edit_telegram_message(chat_id: int, message_id: int, text: str) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not configured')
    url = f'{TELEGRAM_API_URL}/bot{TELEGRAM_BOT_TOKEN}/editMessageText'
    payload = {'chat_id': chat_id, 'message_id': message_id, 'text': text, 'parse_mode': 'Markdown'}
    resp = requests.post(url, json=payload, timeout=15)
    res_json = resp.json()
    
    # Fallback to plain text if Telegram rejects Markdown formatting (e.g. unescaped underscores)
    if not res_json.get('ok'):
        payload.pop('parse_mode', None)
        resp = requests.post(url, json=payload, timeout=15)
        res_json = resp.json()

    return res_json


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
   Memindai seluruh pasar BEI (960+ saham) & menampilkan daftar 5–15 saham yang beraktivitas di luar kebiasaan.

2️⃣ */status* atau */progress*:
   Mengecek status & persentase progres pemindaian real-time saat scan sedang berlangsung.

3️⃣ *Kirim Kode Saham* (misal: `BBRI` atau `ASII TLKM`):
   Melihat fakta observasi deskriptif (Volume percentile, RSI percentile, EMA trend, Breakout, Rarity 12 bulan) khusus saham tersebut.

4️⃣ */dividends*:
   Melihat sinyal aktif & mendatang dari strategi *Dividend Cum-Date Drift* (strategi yang lolos validasi out-of-sample).

5️⃣ */ping*:
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


# ─── Preload State ───────────────────────────────────────────────────────────
PRELOAD_RUNNING = False
PRELOAD_LOCK    = threading.Lock()

# ─── Async Workers (Anti-Timeout & Anti-Stuck) ───────────────────────────────

def _split_long_message(text: str, max_len: int = 4000) -> list[str]:
    """Split long message at line boundaries to preserve Markdown formatting."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find last newline before max_len to avoid breaking mid-line
        split_at = text.rfind('\n', 0, max_len)
        if split_at <= 0:
            # Fallback: find last space
            split_at = text.rfind(' ', 0, max_len)
        if split_at <= 0:
            # Last resort: hard cut at max_len
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks


def _send_long_message(chat_id: int, text: str, max_len: int = 4000):
    """Kirim pesan panjang ke Telegram, potong di batas baris untuk preserve Markdown."""
    for chunk in _split_long_message(text, max_len):
        send_telegram_message(chat_id, chunk)


def _async_run_scan(chat_id: int, msg_id: int):
    """Background worker untuk menjalankan scan tanpa menahan HTTP request Telegram."""
    try:
        send_chat_action(chat_id, 'typing')
        results = scanner.scan_unusual_activity(max_results=10)
        print(f"[Scan] Selesai! {len(results)} saham unusual ditemukan.")
        report = scanner.format_telegram_report(results)
        print(f"[Scan] Report length: {len(report)} chars")

        # Persist scan results ke DB untuk audit trail
        try:
            from datetime import datetime
            scan_date = datetime.now().strftime('%Y-%m-%d')
            saved = db.save_scan_results(scan_date, results)
            print(f"[Scan] {saved} results persisted ke database.")
        except Exception as persist_err:
            print(f"[Scan] Warning: Gagal persist results: {persist_err}")

        # Selalu kirim sebagai pesan BARU di bawah agar tidak pernah hilang
        _send_long_message(chat_id, report)
        
        # Edit pesan loading lama menjadi ringkasan singkat
        if msg_id:
            summary = f"✅ *Scan selesai!* {len(results)} saham unusual ditemukan. Lihat hasil di bawah ⬇️"
            try:
                edit_telegram_message(chat_id, msg_id, summary)
            except Exception:
                pass  # Tidak masalah kalau edit gagal, laporan sudah terkirim
    except Exception as err:
        tb = traceback.format_exc()
        print(f"[Scan] ❌ ERROR: {err}\n{tb}")
        err_msg = f"❌ *Error saat pemindaian*:\n`{str(err)[:200]}`"
        _send_long_message(chat_id, err_msg)
    finally:
        with ACTIVE_SCANS_LOCK:
            ACTIVE_SCANS.discard(chat_id)


def _async_run_ticker_query(chat_id: int, msg_id: int, tickers: list[str]):
    """Background worker untuk query ticker khusus tanpa menahan HTTP request Telegram."""
    try:
        send_chat_action(chat_id, 'typing')
        results = scanner.scan_unusual_activity(tickers=tickers, max_results=len(tickers))
        
        if results:
            report = scanner.format_telegram_report(results)
        else:
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
                    lines.append(f"• **{t.replace('.JK', '')}** (Rp {close_p:,.0f}): RSI 14 = {rsi_str} — Tidak ada aktivitas di luar kebiasaan hari ini.")
                else:
                    lines.append(f"• **{t}**: Data tidak ditemukan.")
            lines.append(f"\n---\n{scanner.MANDATORY_DISCLAIMER if hasattr(scanner, 'MANDATORY_DISCLAIMER') else ''}")
            report = "\n".join(lines)

        # Selalu kirim pesan baru + edit loading jadi ringkasan
        _send_long_message(chat_id, report)
        if msg_id:
            try:
                edit_telegram_message(chat_id, msg_id, "✅ *Data observasi siap. Lihat hasil di bawah ⬇️*")
            except Exception:
                pass
    except Exception as err:
        tb = traceback.format_exc()
        print(f"[Ticker Query] ❌ ERROR: {err}\n{tb}")
        _send_long_message(chat_id, f"❌ *Error saat query ticker*: `{str(err)[:200]}`")
    finally:
        with ACTIVE_SCANS_LOCK:
            ACTIVE_SCANS.discard(chat_id)


def _async_preload_all(chat_id: int, msg_id: int):
    """Background worker untuk mengunduh seluruh data saham BEI secara bertahap."""
    global PRELOAD_RUNNING
    try:
        from data.idx_universe import fetch_live_idx_tickers
        from data.ingestion import compute_indicators

        all_tickers = fetch_live_idx_tickers()
        cached      = set(db.get_tickers())
        to_fetch    = [t for t in all_tickers if t not in cached]

        total   = len(to_fetch)
        done    = 0
        failed  = 0

        if total == 0:
            msg = f"✅ *Preload selesai!* Semua {len(all_tickers)} saham BEI sudah ada di cache."
            edit_telegram_message(chat_id, msg_id, msg)
            return

        edit_telegram_message(
            chat_id, msg_id,
            f"🔄 *Preload dimulai!*\n\nMengunduh `{total}` saham baru dari BEI...\nProses ini berjalan di background. Laporan progres dikirim setiap 50 saham."
        )

        for ticker in to_fetch:
            try:
                df_p = scanner.provider.get_or_fetch(ticker, period_days=252*2)
                if not df_p.empty:
                    df_ind = compute_indicators(df_p)
                    db.save_indicators(ticker, df_ind)
            except Exception:
                failed += 1

            done += 1

            # Kirim progres setiap 50 saham
            if done % 50 == 0 or done == total:
                pct = round(done / total * 100, 1)
                cached_now = len(db.get_tickers())
                try:
                    edit_telegram_message(
                        chat_id, msg_id,
                        f"🔄 *Preload Progres: {done}/{total} saham ({pct}%)*\n"
                        f"• Total di-cache : `{cached_now} saham`\n"
                        f"• Gagal diunduh  : `{failed} saham`\n"
                        f"_{'Selesai! ✅' if done == total else 'Sedang berjalan...'}_"
                    )
                except Exception:
                    pass

        final_cached = len(db.get_tickers())
        send_telegram_message(
            chat_id,
            f"✅ *Preload Selesai!*\n\n"
            f"• Total saham di-cache : `{final_cached} saham`\n"
            f"• Berhasil diunduh     : `{done - failed} saham`\n"
            f"• Gagal diunduh       : `{failed} saham`\n\n"
            f"Sekarang `/scan` akan memindai seluruh `{final_cached}` saham secara instan!"
        )
    except Exception as err:
        tb = traceback.format_exc()
        print(f"[Preload] ❌ ERROR: {err}\n{tb}")
        send_telegram_message(chat_id, f"❌ *Preload error*: `{str(err)[:200]}`")
    finally:
        with PRELOAD_LOCK:
            PRELOAD_RUNNING = False


# ─── Flask Routes ─────────────────────────────────────────────────────────────

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

    # Persist scan results
    try:
        from datetime import datetime
        scan_date = datetime.now().strftime('%Y-%m-%d')
        db.save_scan_results(scan_date, results)
    except Exception:
        pass

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

    # Handle /status or /progress command (Live Scan Progress Tracker)
    if text_lower in ['/status', '/progress', 'progres', 'status', 'progress']:
        send_chat_action(chat_id, 'typing')
        prog = scanner.get_scan_progress()
        if prog.get('is_running'):
            pct_str = f"{prog['pct']}%"
            est_str = f"~{prog['est_remaining_sec']} detik" if prog['est_remaining_sec'] > 0 else "hampir selesai"
            last_t  = prog['last_ticker'].replace('.JK', '') if prog['last_ticker'] else '-'
            msg = (
                f"⏳ *Status Progres Pemindaian Real-Time:*\n\n"
                f"• **Saham Diproses** : `{prog['scanned']} / {prog['total']} saham` ({pct_str})\n"
                f"• **Kondisi Unusual** : `{prog['unusual_found']} ditemukan`\n"
                f"• **Saham Terakhir**  : `{last_t}`\n"
                f"• **Waktu Berjalan**  : `{prog['elapsed_sec']} detik`\n"
                f"• **Estimasi Sisa**   : `{est_str}`\n\n"
                f"_Ketik /status lagi kapan saja untuk memperbarui progres._"
            )
        else:
            msg = "ℹ️ Tidak ada pemindaian pasar yang sedang berjalan saat ini.\n\nKetik /scan untuk memulai pemindaian pasar."
        send_telegram_message(chat_id, msg)
        return jsonify({'ok': True})

    # SAFETY NET CHECK — Mencegah spam scan ganda jika pemindaian sedang berjalan
    with ACTIVE_SCANS_LOCK:
        if chat_id in ACTIVE_SCANS:
            send_telegram_message(
                chat_id,
                "⚠️ *Pemindaian sedang berjalan di background...*\n"
                "_Mohon tunggu hasil pemindaian sebelumnya selesai atau ketik /status untuk mengecek progres._"
            )
            return jsonify({'ok': True})

    # Handle /preload command — Unduh semua saham BEI ke cache sekaligus
    if text_lower in ['/preload', 'preload']:
        global PRELOAD_RUNNING
        with PRELOAD_LOCK:
            if PRELOAD_RUNNING:
                send_telegram_message(chat_id, "⚠️ *Preload sudah berjalan!*\nKetik /status untuk mengecek progres scan, atau tunggu preload selesai.")
                return jsonify({'ok': True})
            PRELOAD_RUNNING = True

        load_res = send_telegram_message(
            chat_id,
            "🔄 *Memulai Preload seluruh saham BEI...*\n_Mengambil daftar saham terbaru dari idx.co.id..._"
        )
        msg_id_preload = load_res.get('result', {}).get('message_id')
        threading.Thread(target=_async_preload_all, args=(chat_id, msg_id_preload), daemon=True).start()
        return jsonify({'ok': True})

    # Handle /scan command (ASYNC BACKGROUND THREAD)
    if text_lower in ['/scan', 'scan', 'scan hari ini']:
        with ACTIVE_SCANS_LOCK:
            ACTIVE_SCANS.add(chat_id)

        send_chat_action(chat_id, 'typing')
        
        load_res = send_telegram_message(
            chat_id,
            "⏳ *Sedang memindai seluruh 960+ saham pasar BEI...*\n"
            "_Pemindaian berjalan di background. Ketik /status kapan saja untuk mengecek progres._"
        )
        msg_id = load_res.get('result', {}).get('message_id')

        threading.Thread(target=_async_run_scan, args=(chat_id, msg_id), daemon=True).start()
        return jsonify({'ok': True})

    # Handle /dividends command
    if text_lower in ['/dividends', 'dividend', 'dividen']:
        send_chat_action(chat_id, 'typing')
        
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

        send_telegram_message(chat_id, msg)
        return jsonify({'ok': True})

    # Handle specific Tickers input (ASYNC BACKGROUND THREAD)
    tickers = parse_tickers_from_text(text)
    if not tickers:
        send_telegram_message(chat_id, '❓ Perintah atau kode saham tidak dikenali.\n\nGunakan /scan, /status, /ping, /dividends, atau ketik kode saham (misal: `BBRI ASII`).')
        return jsonify({'ok': True})

    with ACTIVE_SCANS_LOCK:
        ACTIVE_SCANS.add(chat_id)

    send_chat_action(chat_id, 'typing')
    ticker_str = ", ".join([t.replace('.JK', '') for t in tickers])
    load_res = send_telegram_message(
        chat_id,
        f"⏳ *Sedang memproses & mengambil data observasi {ticker_str}...*"
    )
    msg_id = load_res.get('result', {}).get('message_id')

    threading.Thread(target=_async_run_ticker_query, args=(chat_id, msg_id, tickers), daemon=True).start()
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
