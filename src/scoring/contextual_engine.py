import pandas as pd
from datetime import datetime
from typing import Dict, Any

class ContextualEngine:
    """
    Scoring Engine V2: Menggunakan fitur kontekstual/cross-sectional.
    
    Mendukung dua mode standalone:
    - 'rel_strength': Gunakan rel_strength_5d_rank (Tahap 1)
    - 'vol_accum'   : Gunakan vol_accum_5d_rank (Tahap 2 standalone)
    """
    def __init__(
        self,
        rank_threshold: float = 10.0,
        min_turnover: float = 1_000_000_000,
        feature_mode: str = 'rel_strength',  # 'rel_strength' | 'vol_accum'
    ):
        self.version = f'context_v1.0_{feature_mode}'
        self.min_turnover = min_turnover
        self.rank_threshold = rank_threshold
        self.feature_mode = feature_mode
        
        # Kolom rank yang dipakai sesuai mode
        self._rank_col = {
            'rel_strength': 'rel_strength_5d_rank',
            'vol_accum': 'vol_accum_5d_rank',
        }.get(feature_mode, 'rel_strength_5d_rank')

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
            
        # 2. Extract Contextual Feature (sesuai mode)
        rel_rank = row.get(self._rank_col)
        
        if pd.isna(rel_rank):
            return self._empty_result(ticker, today, f"Missing {self._rank_col}")
            
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


class ContextualEngineAND:
    """
    Tahap 2b: AND Intersection Engine.

    Sinyal BULLISH hanya jika KEDUA kondisi terpenuhi serentak:
      - rel_strength_5d_rank <= rank_rel_strength (paling jeblok vs IHSG)
      - vol_accum_5d_rank    <= rank_vol_accum    (net selling pressure)

    Rationale:
    - Korelasi antar dua rank = 0.59 (moderat, tidak fully independent)
    - Sanity check: di threshold 10-25%, AND menghasilkan 174-645 sinyal/fold (reliable)
    - Dua threshold bisa beda biar Optuna bebas eksplor asimetri optimal

    Syarat mutlak (tidak bisa di-tune):
    - Turnover 5-day avg >= 1 Miliar Rupiah (Microstructure Guard)
    """
    def __init__(
        self,
        rank_rel_strength: float = 15.0,
        rank_vol_accum: float = 15.0,
        min_turnover: float = 1_000_000_000,
    ):
        self.version = 'context_v2b_AND'
        self.rank_rel_strength = rank_rel_strength
        self.rank_vol_accum = rank_vol_accum
        self.min_turnover = min_turnover

    def score(self, ticker: str, df: pd.DataFrame, today: str = None) -> Dict[str, Any]:
        if today is None:
            today = df.index[-1].strftime('%Y-%m-%d')

        today_dt = pd.Timestamp(today)
        if today_dt not in df.index:
            return self._empty_result(ticker, today, "Data tidak tersedia")

        row = df.loc[today_dt]

        # 1. Turnover Filter (tidak bisa di-tune)
        turnover = row.get('turnover_5d', 0)
        if pd.isna(turnover) or turnover < self.min_turnover:
            return self._empty_result(ticker, today, f"Illiquid: turnover < 1M")

        # 2. Extract kedua rank
        rs_rank  = row.get('rel_strength_5d_rank')
        vol_rank = row.get('vol_accum_5d_rank')

        if pd.isna(rs_rank) or pd.isna(vol_rank):
            return self._empty_result(ticker, today, "Missing rank data")

        # 3. AND Logic — keduanya harus masuk kuantil ekstrem
        rs_ok  = rs_rank  <= self.rank_rel_strength
        vol_ok = vol_rank <= self.rank_vol_accum

        breakdown = [
            f"rel_strength_rank={rs_rank:.1f}% ({'✓' if rs_ok else '✗'} <= {self.rank_rel_strength:.1f}%)",
            f"vol_accum_rank={vol_rank:.1f}% ({'✓' if vol_ok else '✗'} <= {self.rank_vol_accum:.1f}%)",
        ]

        if rs_ok and vol_ok:
            skor   = 100
            sinyal = 'BULLISH'
        else:
            skor   = 0
            sinyal = 'NEUTRAL'

        return {
            'kode': ticker,
            'tanggal': today,
            'skor_total': skor,
            'sinyal': sinyal,
            'breakdown': breakdown,
            'scoring_version': self.version,
        }

    def _empty_result(self, ticker: str, today: str, reason: str) -> Dict[str, Any]:
        return {
            'kode': ticker,
            'tanggal': today,
            'skor_total': 0,
            'sinyal': 'NEUTRAL',
            'breakdown': [reason],
            'scoring_version': self.version,
        }

