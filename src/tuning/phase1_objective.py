import optuna
import os
import sys
from pathlib import Path
from datetime import date
import contextlib

_HERE = Path(__file__).parent
_SRC = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from scoring.engine import ScoringEngine
from backtest.engine import WalkForwardBacktester
from backtest.report import BacktestReporter
from features.indicators import ALL_SCORERS

def objective(trial: optuna.Trial, db_path: str) -> float:
    # 1. Tuning Thresholds
    config = {
        'rsi_bullish_threshold': trial.suggest_int('rsi_bullish_threshold', 45, 60),
        'rsi_overbought_threshold': trial.suggest_int('rsi_overbought_threshold', 65, 85),
        'macd_cross_window': trial.suggest_int('macd_cross_window', 1, 5),
        'bullish_threshold': trial.suggest_int('bullish_threshold', 60, 75),
    }
    
    # 2. Tuning Bobot (Weights)
    # Total bobot harus ~100. Kita tuning 6 indikator, indikator ke-7 jadi sisa (atau normalisasi).
    # Berdasarkan diagnostic, kita kasih range berbeda:
    w_ema = trial.suggest_int('w_ema', 15, 45)
    w_sr = trial.suggest_int('w_sr', 15, 45)
    
    w_rsi = trial.suggest_int('w_rsi', 5, 25)
    w_macd = trial.suggest_int('w_macd', 5, 25)
    
    # Noise indicators (kecilkan range)
    w_vol = trial.suggest_int('w_vol', 0, 10)
    w_bb = trial.suggest_int('w_bb', 0, 10)
    w_candle = trial.suggest_int('w_candle', 0, 10)
    
    weights = {
        'EMA Cross': w_ema,
        'Support/Resistance': w_sr,
        'RSI (Mean-Revert)': w_rsi,
        'MACD (Graded Momentum)': w_macd,
        'Volume (RVOL)': w_vol,
        'Bollinger Band': w_bb,
        'Candlestick': w_candle
    }
    
    # Normalisasi bobot agar total persis 100
    total_w = sum(weights.values())
    if total_w == 0:
        return -999.0
        
    normalized_weights = {k: int((v / total_w) * 100) for k, v in weights.items()}
    
    # Pastikan total persis 100 (karena pembulatan int)
    diff = 100 - sum(normalized_weights.values())
    normalized_weights['EMA Cross'] += diff # Tambahkan sisa ke EMA Cross
    
    # 3. Inject Weights ke Scorers
    scorers = []
    # Instantiate ulang biar aman antar trial
    from features.indicators import (
        EMACrossScorer, SupportResistanceScorer, RSIScorer, 
        MACDScorer, VolumeScorer, BollingerScorer, CandlestickScorer
    )
    
    instances = [
        EMACrossScorer(), SupportResistanceScorer(), RSIScorer(),
        MACDScorer(), VolumeScorer(), BollingerScorer(), CandlestickScorer()
    ]
    
    for s in instances:
        # Override class attribute for this instance
        s.max_score = normalized_weights[s.name]
        scorers.append(s)
        
    # 4. Setup Engine & Backtester
    engine = ScoringEngine(scorers=scorers, config=config)
    from data.database import DatabaseManager
    db = DatabaseManager(db_path)
    tickers = db.get_tickers()
    
    # Train/Validation Cutoff = 2026-02-09 (6 months before end date 2026-08-07)
    backtester = WalkForwardBacktester(db, engine, max_date='2026-02-09')
    with contextlib.redirect_stdout(None):
        results = backtester.run(tickers=tickers)
        
    if not results:
        return -999.0
        
    # 5. Hitung Objective Metric
    # Kita cari EV Net rata-rata di seluruh fold
    fold_evs = [f.expected_value('n3') for f in results.folds]
    # Filter None (kalau sample < 10 di satu fold)
    valid_evs = [ev for ev in fold_evs if ev is not None]
    
    if len(valid_evs) < len(results.folds):
        return -999.0
        
    avg_ev = sum(valid_evs) / len(valid_evs)
    
    # Penalti kalau sample size kekecilan (overfitting ke sedikit sinyal)
    if results.total_signals < 50: # Terlalu sedikit sinyal dalam 3 tahun
        return -999.0
        
    # Penalti kalau ada fold yang EV-nya hancur banget (ga robust)
    min_ev = min(valid_evs)
    
    # Objective = Rata-rata EV + (0.5 * Min EV) -> Biar ngangkat fold terburuk juga
    return avg_ev + (0.5 * min_ev)

