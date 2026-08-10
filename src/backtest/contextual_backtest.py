"""
contextual_backtest.py — Walk-forward backtest runner khusus untuk ContextualEngine V2.

Berbeda dari backtester lama yang pakai ScoringEngine + daily_indicators,
runner ini:
1. Ambil data dari get_prices_with_context() (join contextual_indicators + market_index).
2. Menggunakan ContextualEngine sebagai scorer.
3. Menghasilkan BacktestResult yang kompatibel dengan laporan & gate check yang sudah ada.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from data.database import DatabaseManager
from backtest.engine import (
    WalkForwardBacktester, BacktestResult, FoldResult, SignalRecord,
    IDX_ROUNDTRIP_COST
)
from scoring.contextual_engine import ContextualEngine
from scoring.config import SCORING_CONFIG


class ContextualWalkForwardBacktester:
    """
    Walk-forward backtester untuk ContextualEngine.
    Mengoverride satu method kunci _score_one_day dan _load_data
    agar menggunakan konteks data (get_prices_with_context).
    """
    def __init__(
        self,
        db: DatabaseManager,
        engine: ContextualEngine = None,
        n_folds: int = 5,
        min_lookback: int = 60,
        min_score_to_signal: int = 80,
        target_threshold_pct: float = None,
        max_date: str = None,
    ):
        self.db = db
        self.engine = engine or ContextualEngine()
        self.n_folds = n_folds
        self.min_lookback = min_lookback
        self.min_score_to_signal = min_score_to_signal
        self.target_threshold_pct = target_threshold_pct or SCORING_CONFIG['signal_return_threshold']
        self.max_date = max_date

    def run(self, tickers: list[str]) -> BacktestResult:
        print(f"\n{'='*65}")
        print(f"  ContextualEngine V2 — Walk-Forward Backtest")
        print(f"  Tickers  : {len(tickers)}")
        print(f"  Folds    : {self.n_folds}")
        print(f"  Engine   : {self.engine.version}")
        print(f"  Target   : {self.target_threshold_pct}% (net+fee: {self.target_threshold_pct + IDX_ROUNDTRIP_COST*100:.1f}%)")
        print(f"  Holdout  : {'EXCLUDED (max_date=' + self.max_date + ')' if self.max_date else 'INCLUDED'}")
        print(f"{'='*65}")

        all_dates = self._get_all_dates(tickers)
        if len(all_dates) < self.min_lookback + 30:
            raise ValueError(f"Data tidak cukup: {len(all_dates)} hari")

        fold_windows = self._make_fold_windows(all_dates)
        result = BacktestResult(
            tickers=tickers,
            n_folds=self.n_folds,
            target_threshold_pct=self.target_threshold_pct,
        )

        for fold_idx, (test_start, test_end) in enumerate(fold_windows, 1):
            print(f"\n  Fold {fold_idx}/{self.n_folds}: Test [{test_start} → {test_end}]")
            fold_result = self._run_fold(fold_idx, tickers, all_dates, test_start, test_end)
            result.folds.append(fold_result)
            print(f"    Sinyal BULLISH : {len(fold_result.bullish_signals)}")
            ev = fold_result.expected_value()
            wr = fold_result.win_rate()
            if ev is not None:
                print(f"    EV (net fees)  : {ev*100:+.2f}%")
                print(f"    Win Rate       : {wr*100:.1f}%" if wr else "    Win Rate: N/A")
            else:
                print(f"    EV: N/A (sample < 10: {len(fold_result.bullish_signals)} sinyal)")

        result.score_buckets = {}  # ContextualEngine skor biner (0 atau 100), skip bucket breakdown
        print(f"\n{'='*65}")
        print(f"  SELESAI")
        return result

    def _run_fold(self, fold_idx, tickers, all_dates, test_start, test_end) -> FoldResult:
        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=all_dates[0],
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
        )
        test_dates = [d for d in all_dates if test_start <= d <= test_end]

        for ticker in tickers:
            extra_end = self._offset_date(test_end, 10)
            df_full = self.db.get_prices_with_context(ticker, end=extra_end)

            if df_full.empty or len(df_full) < self.min_lookback:
                continue

            df_full.index = pd.to_datetime(df_full.index)

            for test_date in test_dates:
                try:
                    signal = self._score_one_day(ticker, df_full, test_date)
                    if signal:
                        fold.signals.append(signal)
                except Exception:
                    pass

        return fold

    def _score_one_day(self, ticker, df_full, target_date) -> Optional[SignalRecord]:
        target_dt = pd.Timestamp(target_date)
        df_until_today = df_full[df_full.index <= target_dt]
        if len(df_until_today) < self.min_lookback:
            return None

        score_result = self.engine.score(ticker, df_until_today, today=target_date)
        score = score_result['skor_total']
        signal_label = score_result['sinyal']

        if score < self.min_score_to_signal:
            return None

        row_today = df_until_today[df_until_today.index == target_dt]
        if row_today.empty:
            return None
        entry_price = float(row_today['Close'].iloc[0])

        df_after = df_full[df_full.index > target_dt].sort_index()

        def get_exit(n: int):
            return float(df_after['Close'].iloc[n - 1]) if len(df_after) >= n else None

        def calc_return(ep):
            return (ep - entry_price) / entry_price if ep and entry_price > 0 else None

        exit_n1, exit_n3, exit_n5 = get_exit(1), get_exit(3), get_exit(5)
        ret_n1, ret_n3, ret_n5 = calc_return(exit_n1), calc_return(exit_n3), calc_return(exit_n5)
        threshold = self.target_threshold_pct / 100

        return SignalRecord(
            ticker=ticker, date=target_date, score=score, signal=signal_label,
            entry_price=entry_price,
            exit_price_n1=exit_n1, exit_price_n3=exit_n3, exit_price_n5=exit_n5,
            return_n1=ret_n1, return_n3=ret_n3, return_n5=ret_n5,
            hit_target=ret_n3 >= threshold if ret_n3 is not None else None,
            hit_target_net=ret_n3 >= (threshold + IDX_ROUNDTRIP_COST) if ret_n3 is not None else None,
        )

    def _get_all_dates(self, tickers):
        with self.db._connect() as conn:
            placeholders = ','.join('?' * len(tickers))
            query = f"SELECT DISTINCT date FROM daily_prices WHERE ticker IN ({placeholders}) AND is_valid=1"
            params = list(tickers)
            if self.max_date:
                query += " AND date <= ?"
                params.append(self.max_date)
            query += " ORDER BY date ASC"
            rows = conn.execute(query, params).fetchall()
        return [r[0] for r in rows]

    def _make_fold_windows(self, all_dates):
        usable = all_dates[self.min_lookback:]
        fold_size = len(usable) // self.n_folds
        windows = []
        for i in range(self.n_folds):
            s = i * fold_size
            e = s + fold_size - 1
            if i == self.n_folds - 1:
                e = len(usable) - 1
            windows.append((usable[s], usable[e]))
        return windows

    @staticmethod
    def _offset_date(date_str, days):
        dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=days)
        return dt.strftime('%Y-%m-%d')


def main():
    import json

    db = DatabaseManager(str(_ROOT / 'data' / 'idx_screener.db'))
    tickers = db.get_tickers()
    if not tickers:
        print("Tidak ada ticker di DB.")
        return

    # Tahap 1 MVP: hanya rel_strength_5d, holdout excluded
    engine = ContextualEngine()
    backtester = ContextualWalkForwardBacktester(
        db=db,
        engine=engine,
        n_folds=5,
        max_date='2026-02-09',  # Holdout: Feb-Aug 2026 dikunci
    )

    result = backtester.run(tickers)
    metrics = result.aggregate_metrics(horizon='n3')

    print("\n" + "="*65)
    print("  AGGREGATE METRICS (Training Set, N+3)")
    print("="*65)
    for k, v in metrics.items():
        print(f"  {k:30}: {v}")

    gate = metrics.get('lolos_gate', {})
    print("\n  === GATE CHECK ===")
    for g, ok in gate.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {g}: {ok}")

if __name__ == '__main__':
    main()
