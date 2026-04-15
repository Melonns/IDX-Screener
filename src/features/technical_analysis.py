import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator

def add_technical_indicators(df):
    """
    Add advanced technical and momentum indicators.
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
        
    # 5. Volume Analysis & RVOL (Relative Volume)
    df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()
    # RVOL is current volume vs 20-day average. RVOL > 2 is a strong spark.
    df['RVOL'] = df['Volume'] / df['VOL_SMA_20'].replace(0, np.nan)
    
    # 6. Price Density / Volatility (True Range)
    # Estimate True Range (High - Low vs Prev Close)
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['True_Range'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    df['Volatility_Ratio'] = df['True_Range'] / df['ATR_14'].replace(0, np.nan)
    
    # 7. Gap Up Indicator (Did Open jump higher than yesterday's High?)
    prev_high = df['High'].shift(1)
    df['Gap_Up'] = np.where(df['Open'] > prev_high, 1, 0)
    
    # 8. Core Momentum
    df['Pct_Change'] = df['Close'].pct_change() * 100
    
    return df

def create_momentum_label(df, threshold=5.0):
    """
    Create binary target label (Target_ARA) shifted by T-1.
    If TOMORROW's Return (High vs Today's Close) is >= threshold, 
    then TODAY gets labeled as 1.
    """
    if df.empty:
        return df
        
    # Predict maximum achievable return tomorrow (High_t+1 / Close_t - 1)
    tomorrow_high = df['High'].shift(-1)
    next_day_max_return = ((tomorrow_high - df['Close']) / df['Close']) * 100
    
    # Create the label: 1 if hit target, 0 otherwise
    df['Target_Momentum'] = np.where(next_day_max_return >= threshold, 1, 0)
    
    # Remove the very last row since it won't have a label for "tomorrow"
    df.loc[df.index[-1], 'Target_Momentum'] = np.nan
    
    return df

def feature_engineer(filepath):
    """
    Load raw data, apply indicators and labels.
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    df = create_momentum_label(df, threshold=5.0)
    return df
