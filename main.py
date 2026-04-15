import io
import json
import os
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
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
    df = add_technical_indicators(df)
    df = create_momentum_label(df)
    df = df.dropna(subset=selected_features)
    return df


@app.route('/')
def status():
    return jsonify({
        'status': 'ok',
        'model_file': MODEL_FILE,
        'model_url': MODEL_URL,
        'selected_features_count': len(selected_features),
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
    except Exception as err:
        return jsonify({'error': f'Failed to load data: {err}'}), 400

    if df.empty:
        return jsonify({'error': 'No valid rows available after feature engineering.'}), 400

    df = df.dropna(subset=selected_features)
    if df.empty:
        return jsonify({'error': 'No rows with complete selected features.'}), 400

    row = df.iloc[[-1]]
    X = row[selected_features]
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0].tolist()

    return jsonify({
        'prediction': pred,
        'probability': prob,
        'selected_features': selected_features,
        'timestamp': str(row.index[-1]),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
