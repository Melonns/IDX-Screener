import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator

def add_technical_indicators(df):
    """
    Add basic technical indicators to a pandas DataFrame containing OHLCV data.
    Assumes standard yfinance columns: Open, High, Low, Close, Volume.
    """
    if df.empty or len(df) < 50:
        return df
    
    # 1. Moving Averages
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['EMA_200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
    
    # 2. RSI (Relative Strength Index)
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # 3. MACD
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
        
    # 4. Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
        
    # 5. Volume Analysis (Volume moving average)
    df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # 6. Custom ARA Potential Features
    # Volume spike: Volume is X times higher than 20-day average
    df['Volume_Spike'] = df['Volume'] / df['VOL_SMA_20']
    
    # Price Momentum: Percent change
    df['Pct_Change'] = df['Close'].pct_change() * 100
    
    return df

def feature_engineer(filepath):
    """
    Load raw data, apply indicators, and return processed dataframe.
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    return df
