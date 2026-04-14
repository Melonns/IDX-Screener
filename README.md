# IDX Stock Screener

A Machine Learning project for analyzing and screening Indonesian Stock Exchange (IDX) stocks. The aim is to identify stocks with high potential (e.g., potential auto reject atas / ARA) using technical analysis, historical price data, and volume spread analysis.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Melonns/IDX-Screener.git
   cd IDX-Screener
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
- `data/` : Contains `raw` (unmodified data) and `processed` (feature engineered data) folders.
- `src/` : Main source code for the project.
  - `src/data/` : Scripts to fetch data (e.g., via yfinance).
  - `src/features/` : Technical indicators and feature engineering.
  - `src/models/` : Machine learning model pipelines.
- `notebooks/` : Jupyter Notebooks for data exploration and backtesting.
- `models/` : Saved/serialized trained ML models.

## Usage
Run the ingestion script to get the latest data:
```bash
python src/data/ingest.py
```
