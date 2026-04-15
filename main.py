import io
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, request

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from features.technical_analysis import add_technical_indicators, create_momentum_label

app = Flask(__name__)

MODEL_FILE = os.environ.get(
    'MODEL_FILE', 'models/xgboost_momentum_20260415_0856.joblib'
)
MODEL_URL = os.environ.get(
    'MODEL_URL', 'https://drive.google.com/uc?export=download&id=1WIiOzQY9lHl3JdeOWQKj8wWHGFKtqaS1'
)
FEATURE_FILE = os.environ.get(
    'FEATURE_FILE', 'models/selected_features_20260415_0856.json'
)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')

TELEGRAM_API_URL = 'https://api.telegram.org'


def get_google_drive_file_id(url: str) -> str | None:
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None


def build_drive_download_url(url: str) -> str:
    file_id = get_google_drive_file_id(url)
    if file_id:
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url


def save_response_content(response: requests.Response, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def download_file(url: str, destination: str) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    download_url = build_drive_download_url(url)

    with requests.Session() as session:
        response = session.get(download_url, stream=True)
        if 'confirm=' in response.url or 'download_warning' in response.cookies:
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                response = session.get(
                    'https://drive.google.com/uc',
                    params={'export': 'download', 'id': get_google_drive_file_id(download_url), 'confirm': token},
                    stream=True,
                )

        if response.status_code != 200:
            raise RuntimeError(
                f'Failed to download model from {download_url}: HTTP {response.status_code}'
            )

        save_response_content(response, destination_path)


def ensure_model_available() -> None:
    model_path = Path(MODEL_FILE)
    if model_path.exists():
        return

    if not MODEL_URL:
        raise FileNotFoundError(
            f'Model file {MODEL_FILE} not found and MODEL_URL is not set.'
        )

    print(f'Downloading model from {MODEL_URL} to {MODEL_FILE}...')
    download_file(MODEL_URL, MODEL_FILE)
    print('Model downloaded successfully.')


ensure_model_available()
model = joblib.load(MODEL_FILE)
with open(FEATURE_FILE, 'r', encoding='utf-8') as f:
    selected_features = json.load(f)


def prepare_data_from_csv_text(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=True)
    return prepare_data_from_dataframe(df)


def prepare_data_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_indicators(df)
    df = create_momentum_label(df)
    df = df.dropna(subset=selected_features)
    return df


def download_ticker_data(ticker: str, period: str = '1y', interval: str = '1d', max_retries: int = 3) -> pd.DataFrame:
    """
    Download ticker data from yfinance with retry logic and better error handling.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'ASII.JK')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
        interval: Interval ('1d', '1h', '1m')
        max_retries: Number of retry attempts
    
    Returns:
        DataFrame with OHLCV columns
    
    Raises:
        ValueError: If no data found after retries
    """
    formats_to_try = [ticker]
    
    # If ticker ends with .JK, also try without it
    if ticker.endswith('.JK'):
        formats_to_try.append(ticker[:-3])
    # If ticker doesn't end with .JK but is 4 chars, try with .JK
    elif len(ticker) == 4 and ticker.isalpha():
        formats_to_try.append(f'{ticker}.JK')
    
    last_error = None
    
    for attempt in range(max_retries):
        for fmt in formats_to_try:
            try:
                # Add small delay between retries to avoid rate limiting
                if attempt > 0:
                    time.sleep(1 + attempt * 0.5)
                
                # Download with timeout
                df = yf.download(fmt, period=period, interval=interval, progress=False, timeout=10)
                
                if df.empty:
                    last_error = f"Empty DataFrame for {fmt}"
                    continue
                
                # Handle MultiIndex columns (yfinance quirk for single tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Validate OHLCV columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    last_error = f"Incomplete OHLCV data for {fmt}"
                    continue
                
                # Success!
                return df[required_cols]
                
            except Exception as e:
                last_error = str(e)
                continue
    
    # All retries failed
    raise ValueError(
        f'Failed to download {ticker} after {max_retries} attempts '
        f'(tried formats: {", ".join(formats_to_try)}). '
        f'Last error: {last_error}'
    )




def predict_from_dataframe(df: pd.DataFrame) -> dict:
    df = prepare_data_from_dataframe(df)
    if df.empty:
        raise ValueError('Not enough historical rows to compute all required features.')
    row = df.iloc[[-1]]
    X = row[selected_features]
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0].tolist()
    return {
        'prediction': pred,
        'probability': prob,
        'timestamp': str(row.index[-1]),
    }


def parse_tickers_from_text(text: str) -> list[str]:
    tokens = re.findall(r'[A-Za-z0-9\.]+', text.upper())
    tickers = []
    for token in tokens:
        token = token.strip()
        if token.endswith('JK') and not token.endswith('.JK'):
            token = token[:-2] + '.JK'
        if token.endswith('.JK') or (len(token) == 4 and token.isalpha()):
            tickers.append(token)
    return list(dict.fromkeys(tickers))


def send_telegram_message(chat_id: int, text: str) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not configured')
    url = f'{TELEGRAM_API_URL}/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}
    resp = requests.post(url, json=payload, timeout=15)
    return resp.json()


def send_chat_action(chat_id: int, action: str = 'typing') -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is not configured')
    url = f'{TELEGRAM_API_URL}/bot{TELEGRAM_BOT_TOKEN}/sendChatAction'
    payload = {'chat_id': chat_id, 'action': action}
    resp = requests.post(url, json=payload, timeout=15)
    return resp.json()


@app.route('/')
def status():
    return jsonify({
        'status': 'ok',
        'model_file': MODEL_FILE,
        'model_url': MODEL_URL,
        'selected_features_count': len(selected_features),
        'telegram_bot': bool(TELEGRAM_BOT_TOKEN),
    })


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    csv_text = payload.get('csv_text')
    csv_url = payload.get('csv_url')

    if not csv_text and not csv_url:
        return jsonify({'error': 'Provide csv_text or csv_url in JSON body.'}), 400

    try:
        if csv_url:
            df = pd.read_csv(csv_url, index_col=0, parse_dates=True)
        else:
            df = prepare_data_from_csv_text(csv_text)
        result = predict_from_dataframe(df)
    except Exception as err:
        return jsonify({'error': f'Failed to load data: {err}'}), 400

    return jsonify({
        **result,
        'selected_features': selected_features,
    })


@app.route('/predict_tickers', methods=['POST'])
def predict_tickers():
    payload = request.get_json(force=True)
    tickers = payload.get('tickers')

    if not tickers or not isinstance(tickers, list):
        return jsonify({'error': 'Provide tickers as a JSON list under key tickers.'}), 400

    results = {}
    for ticker in tickers:
        try:
            df = download_ticker_data(ticker)
            result = predict_from_dataframe(df)
            results[ticker] = {
                'status': 'ok',
                'prediction': result['prediction'],
                'probability': result['probability'],
                'timestamp': result['timestamp'],
            }
        except Exception as err:
            results[ticker] = {'status': 'error', 'error': str(err)}

    return jsonify(results)


@app.route('/telegram_webhook', methods=['POST'])
def telegram_webhook():
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({'error': 'TELEGRAM_BOT_TOKEN is not configured'}), 500

    data = request.get_json(force=True)
    message = data.get('message') or data.get('edited_message')
    if not message:
        return jsonify({'ok': False, 'error': 'No message received'}), 200

    chat_id = message['chat']['id']
    text = message.get('text', '').strip()
    if not text:
        send_telegram_message(chat_id, 'Kirim kode saham IDX seperti: ASII.JK ADMR.JK atau tanpa .JK: ASII ADMR')
        return jsonify({'ok': True})

    tickers = parse_tickers_from_text(text)
    if not tickers:
        send_telegram_message(chat_id, 'Tidak ada ticker yang valid. Gunakan format seperti ASII.JK atau ADMR.JK')
        return jsonify({'ok': True})

    send_chat_action(chat_id, 'typing')
    send_telegram_message(chat_id, f'⏳ Sedang memproses {len(tickers)} ticker... tunggu sebentar ya.')

    replies = []
    success_count = 0
    
    for ticker in tickers:
        send_chat_action(chat_id, 'typing')
        try:
            df = download_ticker_data(ticker, period='1y')
            result = predict_from_dataframe(df)
            prob = result['probability'][1] if len(result['probability']) > 1 else result['probability'][0]
            
            # Format output
            pred_text = "📈 Bullish" if result['prediction'] == 1 else "📉 Bearish"
            replies.append(f'✅ *{ticker}*\n  {pred_text}, Confidence: {prob:.1%}')
            success_count += 1
            
        except ValueError as err:
            # Ticker not found or no data - user-friendly message
            replies.append(f'❌ *{ticker}*: Data tidak tersedia (kemungkinan ticker delisted)')
        except Exception as err:
            # Other errors - still report but keep short
            error_msg = str(err)[:40]
            replies.append(f'❌ *{ticker}*: Error ({error_msg}...)')

    # Build final message
    if success_count == 0:
        message_text = '❌ Semua ticker tidak ditemukan.\nCoba dengan ticker populer: ASII.JK, ADMR.JK, PGAS.JK'
    else:
        message_text = f'✅ Berhasil process {success_count}/{len(tickers)} ticker:\n\n' + '\n\n'.join(replies)
    
    send_telegram_message(chat_id, message_text)
    return jsonify({'ok': True})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
