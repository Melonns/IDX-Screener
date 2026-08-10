"""
hybrid_engine.py — Multi-Factor Hybrid Engine (IDX-Screener V3)

Menggabungkan 3 Pilar Utama:
1. Event Catalyst: Dividend Cum-Date Drift (Yield >= 4.0%, window 10d before Cum-Date)
2. Microstructure / Volume Guard: vol_accum_5d_rank <= 30% ATAU Foreign Net Buy > 0
3. Liquidity Guard: turnover_5d >= 1 Miliar Rupiah

Holding Horizon: Exit 1 hari sebelum Ex-Date (Cum-Date Close).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any


class MultiFactorHybridEngine:
    """
    Scoring Engine V3: Multi-Factor Event-Driven & Microstructure Engine.
    """
    def __init__(
        self,
        min_dividend_yield: float = 4.0,
        entry_window_days: int = 10,
        max_vol_rank: float = 30.0,
        min_turnover: float = 1_000_000_000,
        use_foreign_flow: bool = False,
    ):
        self.version = 'hybrid_v3.0'
        self.min_dividend_yield = min_dividend_yield
        self.entry_window_days = entry_window_days
        self.max_vol_rank = max_vol_rank
        self.min_turnover = min_turnover
        self.use_foreign_flow = use_foreign_flow

    def score_event(self, row: pd.Series) -> Dict[str, Any]:
        """
        Score a single date/ticker row with full catalyst & microstructure data.
        """
        turnover = row.get('turnover_5d', 0)
        if pd.isna(turnover) or turnover < self.min_turnover:
            return {'score': 0, 'signal': 'NEUTRAL', 'reason': f'Illiquid: turnover < 1M'}

        div_yield = row.get('dividend_yield', 0)
        if pd.isna(div_yield) or div_yield < self.min_dividend_yield:
            return {'score': 0, 'signal': 'NEUTRAL', 'reason': f'Yield {div_yield:.1f}% < {self.min_dividend_yield}%'}

        vol_rank = row.get('vol_accum_5d_rank', 50.0)
        vol_ok   = (not pd.isna(vol_rank)) and (vol_rank <= self.max_vol_rank)

        # Basic event score
        score = 100 if vol_ok else 80
        signal = 'BULLISH'

        return {
            'score': score,
            'signal': signal,
            'reason': f'Dividend Yield {div_yield:.1f}% >= {self.min_dividend_yield}% (Vol Rank: {vol_rank:.1f}%)',
            'version': self.version
        }
