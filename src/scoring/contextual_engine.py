import pandas as pd
from datetime import datetime
from typing import Dict, Any

class ContextualEngine:
    """
    Scoring Engine V2: Menggunakan fitur kontekstual/cross-sectional.
    
    Tahap 1 MVP:
    - Hanya menggunakan rel_strength_5d_rank.
    - Sinyal beli: Saham berada di 10% terendah rel_strength_5d (paling dibanting relatif IHSG).
    - Syarat mutlak: Turnover rata-rata 5 hari > 1 Miliar Rupiah.
    """
    def __init__(self, rank_threshold: float = 10.0, min_turnover: float = 1_000_000_000):
        self.version = 'context_v1.0'
        self.min_turnover = min_turnover   # Default: 1 Miliar Rupiah
        self.rank_threshold = rank_threshold  # Default: 10% terendah

    def score(self, ticker: str, df: pd.DataFrame, today: str = None) -> Dict[str, Any]:
        """
        Hitung skor untuk satu ticker pada hari tertentu (default hari terakhir di df).
        DataFrame harus berasal dari DatabaseManager.get_prices_with_context().
        """
        if today is None:
            today = df.index[-1].strftime('%Y-%m-%d')
            
        today_dt = pd.Timestamp(today)
        if today_dt not in df.index:
            return self._empty_result(ticker, today, "Data tidak tersedia untuk tanggal ini")
            
        # Ambil row target
        row = df.loc[today_dt]
        
        # 1. Turnover Filter (Microstructure Guard)
        turnover = row.get('turnover_5d', 0)
        if pd.isna(turnover) or turnover < self.min_turnover:
            return self._empty_result(ticker, today, f"Illiquid: Turnover {turnover:,.0f} < 1M")
            
        # 2. Extract Contextual Feature
        rel_rank = row.get('rel_strength_5d_rank')
        
        if pd.isna(rel_rank):
            return self._empty_result(ticker, today, "Missing rel_strength_5d_rank")
            
        # 3. Hitung Skor
        skor = 0
        breakdown = []
        
        # Karena kita mencari saham yang paling oversold relatif,
        # rank yang rendah (mendekati 0) berarti returnnya paling hancur dibanding IHSG.
        # Jika masuk top 10% terburuk (rank <= 10.0), kasih skor maksimal.
        if rel_rank <= self.rank_threshold:
            skor = 100
            breakdown.append(f"Oversold Rank: {rel_rank:.1f}% (<= {self.rank_threshold}%)")
        else:
            skor = 0
            breakdown.append(f"Rank {rel_rank:.1f}% > {self.rank_threshold}%")
            
        # Sinyal
        if skor >= 80:
            sinyal = 'BULLISH'
        elif skor <= 20:
            sinyal = 'BEARISH'
        else:
            sinyal = 'NEUTRAL'
            
        return {
            'kode': ticker,
            'tanggal': today,
            'skor_total': skor,
            'sinyal': sinyal,
            'breakdown': breakdown,
            'scoring_version': self.version
        }
        
    def _empty_result(self, ticker: str, today: str, reason: str) -> Dict[str, Any]:
        return {
            'kode': ticker,
            'tanggal': today,
            'skor_total': 0,
            'sinyal': 'NEUTRAL',
            'breakdown': [reason],
            'scoring_version': self.version
        }
