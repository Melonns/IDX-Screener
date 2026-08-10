import yfinance as yf
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
_HERE = Path(__file__).parent
_SRC = _HERE.parent
sys.path.insert(0, str(_SRC))

from data.database import DatabaseManager

class IndexManager:
    """
    Manages market index data (e.g. ^JKSE) to be used as contextual features.
    Downloads from yfinance and stores in the local SQLite database.
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._init_table()

    def _init_table(self):
        with self.db._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_index (
                    date TEXT PRIMARY KEY,
                    symbol TEXT,
                    close REAL,
                    volume REAL,
                    ret_5d REAL,
                    ret_20d REAL,
                    ma_20 REAL,
                    slope_20d REAL
                )
            """)

    def update_index(self, symbol: str = '^JKSE'):
        print(f"Updating index data for {symbol}...")
        
        # Determine start date
        with self.db._connect() as conn:
            res = conn.execute("SELECT MAX(date) FROM market_index WHERE symbol = ?", (symbol,)).fetchone()
            last_date = res[0] if res and res[0] else None

        if last_date:
            start_date = pd.to_datetime(last_date).strftime('%Y-%m-%d')
        else:
            # If empty, fetch enough history to cover our daily_prices data + lookback buffer
            with self.db._connect() as conn:
                price_min = conn.execute("SELECT MIN(date) FROM daily_prices").fetchone()
                db_min = price_min[0] if price_min and price_min[0] else '2023-01-01'
            start_date = (pd.to_datetime(db_min) - pd.DateOffset(days=50)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date >= end_date:
            print("Index data already up to date.")
            return

        # Fetch from yfinance
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            print("No new index data fetched.")
            return
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = df.index.tz_localize(None)
        
        # Calculate base metrics
        df['ret_5d'] = df['Close'].pct_change(periods=5)
        df['ret_20d'] = df['Close'].pct_change(periods=20)
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['slope_20d'] = (df['ma_20'] - df['ma_20'].shift(5)) / (df['ma_20'].shift(5) + 1e-8)
        
        df.reset_index(inplace=True)
        date_col = 'Date' if 'Date' in df.columns else 'index'
        df['date'] = df[date_col].dt.strftime('%Y-%m-%d')
        df['symbol'] = symbol
        
        # Rename columns to match db schema
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume'})
        
        df_valid = df.dropna(subset=['close', 'ret_5d', 'slope_20d'])
        
        # Insert to DB
        records = df_valid[['date', 'symbol', 'close', 'volume', 'ret_5d', 'ret_20d', 'ma_20', 'slope_20d']].to_dict('records')
        
        with self.db._connect() as conn:
            for r in records:
                conn.execute("""
                    INSERT OR REPLACE INTO market_index 
                    (date, symbol, close, volume, ret_5d, ret_20d, ma_20, slope_20d)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (r['date'], r['symbol'], r['close'], r['volume'], r['ret_5d'], r['ret_20d'], r['ma_20'], r['slope_20d']))
                
        print(f"Inserted/Updated {len(records)} index rows.")

    def get_index_data(self, symbol: str = '^JKSE') -> pd.DataFrame:
        query = "SELECT * FROM market_index WHERE symbol = ? ORDER BY date ASC"
        with self.db._connect() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol,))
        return df

if __name__ == '__main__':
    from pathlib import Path
    _HERE = Path(__file__).parent
    _ROOT = _HERE.parent.parent
    db = DatabaseManager(str(_ROOT / 'data' / 'idx_screener.db'))
    mgr = IndexManager(db)
    mgr.update_index('^JKSE')
