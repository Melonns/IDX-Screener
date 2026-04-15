# IDX Stock Screener

A version 1 machine learning workflow for Indonesian Stock Exchange (IDX) momentum screening. The pipeline ingests daily OHLCV data, computes technical indicators, trains an XGBoost model, and exposes inference through a Flask API and Telegram bot.

## Features
- Ingest raw IDX stock data from Yahoo Finance using dynamic ticker universe selection.
- Compute technical indicators including EMA, RSI, MACD, Bollinger Bands, ATR, RVOL, and gap signals.
- Train an XGBoost classifier on engineered features to predict momentum target labels.
- Serve predictions through REST endpoints and a Telegram webhook bot.
- Support multi-ticker input with space or comma delimiters.
- Provide `/help` and `/example` commands for Telegram users.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Melonns/IDX-Screener.git
   cd IDX-Screener
   ```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Environment Variables
Set these variables before running the Flask app or Telegram webhook:
- `MODEL_FILE` : Local model path, e.g. `models/xgboost_momentum_20260415_0856.joblib`
- `MODEL_URL` : URL to download the model if not present locally
- `FEATURE_FILE` : Feature list JSON file path, e.g. `models/selected_features_20260415_0856.json`
- `TELEGRAM_BOT_TOKEN` : Telegram bot token for webhook replies
- `PORT` : Optional port for the Flask server (default `8080`)

## Project Structure
- `data/raw/` : Raw OHLCV CSV files from yfinance.
- `data/processed/` : Feature-engineered dataset files.
- `models/` : Trained model files and selected feature metadata.
- `src/data/` : Data ingestion and universe selection scripts.
- `src/features/` : Technical indicator and feature engineering logic.
- `src/models/` : Training pipeline and model evaluation scripts.
- `main.py` : Flask API and Telegram webhook server.

## Data Pipeline
1. `src/data/ingest.py` downloads raw OHLCV data for ticker universe.
2. `src/features/technical_analysis.py` computes technical indicators.
3. `src/data/build_dataset.py` builds a processed dataset from raw files.
4. `src/models/train.py` trains the XGBoost classifier and extracts feature importance.

## Running the Project
### Ingest data
```bash
python src/data/ingest.py
```

### Build processed dataset
```bash
python src/data/build_dataset.py
```

### Train model
```bash
python src/models/train.py
```

### Run API server
```bash
python main.py
```

## API Endpoints
- `GET /` : Health check and configuration status.
- `POST /predict` : Submit historical CSV data for prediction.
- `POST /predict_tickers` : Submit a JSON list of tickers for prediction.
- `POST /telegram_webhook` : Telegram bot webhook endpoint.

## Telegram Bot Usage
Send ticker symbols via Telegram, using either spaces or commas:
- `ASII BBCA TLKM`
- `ASII, BBCA, TLKM`
- `ASII.JK BBCA.JK`

Bot commands:
- `/help` : Show usage instructions.
- `/start` : Show welcome and help information.
- `/example` : Show example tickers.

## Notes
- Raw stock files contain only OHLCV data. All technical indicators are generated during feature engineering.
- The current model uses a selected subset of top features for inference.
- For IDX momentum screening, the model predicts whether the next-day high can reach a threshold relative to today close.

## Limitations
- Yahoo Finance is used for price data and may be unreliable on some tickers or environments.
- The current implementation is a proof of concept; data quality and API stability should be monitored.
- Broker flow or order book data are not yet included in this version.

