"""
engine.py — Walk-Forward Backtester untuk IDX-Screener v2.

Cara pakai:
    python -m src.backtest.engine --tickers BBCA ASII TLKM --folds 5
    python -m src.backtest.engine --all-db --folds 5 --min-score 60
    python -m src.backtest.engine --all-db --report report.html

PENTING:
    Backtesting ini adalah pembuktian sesungguhnya — bukan CLI scoring.
    Output CLI yang bagus cuma membuktikan logic jalan tanpa error.
    Baca hasil backtest dengan critical eye:
    - Sample count per score bucket wajib dicek
    - Konsistensi antar fold lebih penting dari aggregate win rate
    - EV net of fees (bukan gross) yang relevan
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import config as app_config
from data.database import DatabaseManager
from data.ingestion import compute_indicators
from scoring.engine import ScoringEngine
from scoring.config import SCORING_CONFIG

# ─── Biaya transaksi IDX (roundtrip) ─────────────────────────────────────────
# Dipakai untuk hitung EV net of fees.
# Sesuaikan dengan sekuritas yang dipakai kalau lebih presisi.
IDX_ROUNDTRIP_COST = 0.004  # 0.4% = ~0.15% beli + 0.25% jual (termasuk PPh)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    """Satu sinyal yang dihasilkan scoring engine pada satu hari."""
    ticker: str
    date: str
    score: int
    signal: str          # BULLISH / NEUTRAL / BEARISH
    entry_price: float
    exit_price_n1: Optional[float] = None
    exit_price_n3: Optional[float] = None
    exit_price_n5: Optional[float] = None
    return_n1: Optional[float] = None
    return_n3: Optional[float] = None
    return_n5: Optional[float] = None
    hit_target: Optional[bool] = None  # return_n3 >= target_threshold (gross)
    hit_target_net: Optional[bool] = None  # return_n3 >= target (setelah fee)


@dataclass
class FoldResult:
    """Hasil backtest satu fold walk-forward."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    signals: list[SignalRecord] = field(default_factory=list)

    @property
    def n_signals(self) -> int:
        return len(self.signals)

    @property
    def bullish_signals(self) -> list[SignalRecord]:
        return [s for s in self.signals if s.signal == 'BULLISH']

    def win_rate(self, horizon: str = 'n3', net_of_fees: bool = True) -> Optional[float]:
        """Win rate untuk horizon tertentu. Return None kalau sample < 5."""
        valid = [s for s in self.bullish_signals
                 if getattr(s, f'return_{horizon}') is not None]
        if len(valid) < 5:
            return None
        threshold = SCORING_CONFIG['signal_return_threshold'] / 100
        if net_of_fees:
            threshold += IDX_ROUNDTRIP_COST
        wins = sum(1 for s in valid if getattr(s, f'return_{horizon}', 0) >= threshold)
        return wins / len(valid)

    def avg_win(self, horizon: str = 'n3') -> Optional[float]:
        """Rata-rata return dari sinyal yang menang."""
        valid = [s for s in self.bullish_signals
                 if getattr(s, f'return_{horizon}') is not None]
        wins = [getattr(s, f'return_{horizon}') for s in valid
                if getattr(s, f'return_{horizon}', 0) >= 0]
        return float(np.mean(wins)) if wins else None

    def avg_loss(self, horizon: str = 'n3') -> Optional[float]:
        """Rata-rata return dari sinyal yang kalah (negatif)."""
        valid = [s for s in self.bullish_signals
                 if getattr(s, f'return_{horizon}') is not None]
        losses = [getattr(s, f'return_{horizon}') for s in valid
                  if getattr(s, f'return_{horizon}', 0) < 0]
        return float(np.mean(losses)) if losses else None

    def expected_value(self, horizon: str = 'n3') -> Optional[float]:
        """
        EV = WR × AvgWin − LR × AvgLoss, net of fees.
        Return None kalau sample tidak cukup (< 10 sinyal bullish).
        """
        valid = [s for s in self.bullish_signals
                 if getattr(s, f'return_{horizon}') is not None]
        if len(valid) < 10:
            return None

        returns = [getattr(s, f'return_{horizon}') for s in valid]
        threshold = SCORING_CONFIG['signal_return_threshold'] / 100
        wins  = [r for r in returns if r >= threshold]
        losses = [r for r in returns if r < threshold]

        wr = len(wins) / len(returns) if returns else 0
        lr = len(losses) / len(returns) if returns else 0
        avg_w = float(np.mean(wins)) if wins else 0
        avg_l = float(np.mean(losses)) if losses else 0

        ev_gross = wr * avg_w + lr * avg_l  # note: avg_l sudah negatif
        ev_net   = ev_gross - IDX_ROUNDTRIP_COST
        return ev_net

    def max_drawdown(self, horizon: str = 'n3') -> Optional[float]:
        """Max loss dari sinyal bullish tunggal (worst case single signal)."""
        valid = [getattr(s, f'return_{horizon}')
                 for s in self.bullish_signals
                 if getattr(s, f'return_{horizon}') is not None]
        return float(min(valid)) if valid else None


@dataclass
class BacktestResult:
    """Hasil keseluruhan walk-forward backtest semua fold."""
    tickers: list[str]
    n_folds: int
    target_threshold_pct: float
    folds: list[FoldResult] = field(default_factory=list)
    score_buckets: dict = field(default_factory=dict)  # per-bucket stats

    def aggregate_metrics(self, horizon: str = 'n3') -> dict:
        """Agregasi metrics semua fold. Return dict lengkap."""
        all_signals = [s for fold in self.folds for s in fold.bullish_signals
                       if getattr(s, f'return_{horizon}') is not None]
        all_returns = [getattr(s, f'return_{horizon}') for s in all_signals]
        threshold   = self.target_threshold_pct / 100

        n = len(all_signals)
        if n == 0:
            return {'error': 'Tidak ada sinyal bullish dengan data return yang cukup'}

        wins   = [r for r in all_returns if r >= threshold]
        losses = [r for r in all_returns if r < threshold]
        wr     = len(wins) / n

        avg_w = float(np.mean(wins))  if wins   else 0.0
        avg_l = float(np.mean(losses)) if losses else 0.0
        ev_gross = wr * avg_w + (1 - wr) * avg_l
        ev_net   = ev_gross - IDX_ROUNDTRIP_COST

        # EV dan WR per fold (untuk cek konsistensi)
        fold_evs  = [f.expected_value(horizon) for f in self.folds]
        fold_wrs  = [f.win_rate(horizon) for f in self.folds]
        fold_evs_valid = [x for x in fold_evs if x is not None]
        fold_wrs_valid = [x for x in fold_wrs if x is not None]

        ev_std    = float(np.std(fold_evs_valid)) if fold_evs_valid else None
        ev_sharpe = float(np.mean(fold_evs_valid) / (np.std(fold_evs_valid) + 1e-6)) \
                    if fold_evs_valid else None

        n_positive_ev_folds = sum(1 for ev in fold_evs_valid if ev > 0)

        return {
            # Aggregate
            'n_signals_bullish': n,
            'n_folds_with_data': len(fold_evs_valid),
            'win_rate': round(wr, 4),
            'avg_win':  round(avg_w, 4),
            'avg_loss': round(avg_l, 4),
            'win_loss_ratio': round(avg_w / abs(avg_l), 2) if avg_l != 0 else None,
            'ev_gross': round(ev_gross, 4),
            'ev_net_of_fees': round(ev_net, 4),
            'max_drawdown': round(min(all_returns), 4) if all_returns else None,
            # Fold consistency
            'ev_per_fold': [round(x, 4) if x else None for x in fold_evs],
            'wr_per_fold': [round(x, 4) if x else None for x in fold_wrs],
            'ev_std_antar_fold': round(ev_std, 4) if ev_std else None,
            'ev_sharpe_like': round(ev_sharpe, 4) if ev_sharpe else None,
            'n_fold_ev_positif': n_positive_ev_folds,
            # Gate check
            'lolos_gate': self._check_gate(wr, avg_w, avg_l, ev_net, n_positive_ev_folds),
        }

    def _check_gate(self, wr, avg_w, avg_l, ev_net, n_positive_ev_folds) -> dict:
        """Cek semua kriteria Phase 1 gate."""
        wl_ratio = abs(avg_w / avg_l) if avg_l != 0 else None
        return {
            'ev_net_ok':        ev_net > 0.003,           # EV > 0.3% net of fees
            'wr_ok':            wr >= 0.55,                # Win rate >= 55%
            'wl_ratio_ok':      wl_ratio >= 1.2 if wl_ratio else False,
            'fold_consistency': n_positive_ev_folds >= 4,  # >= 4 dari 5 fold
            'semua_lolos':      (ev_net > 0.003 and wr >= 0.55
                                 and (wl_ratio >= 1.2 if wl_ratio else False)
                                 and n_positive_ev_folds >= 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Backtester
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward backtester untuk scoring engine IDX-Screener v2.

    Metodologi:
    1. Split data historis menjadi N fold secara kronologis (tidak ada shuffling).
    2. Setiap fold: gunakan data sebelumnya sebagai "training context" (untuk
       kalkulasi indikator yang butuh lookback), lalu score setiap hari di window test.
    3. Setiap sinyal BULLISH: catat entry price (close hari itu) dan exit price
       N hari kemudian (n1, n3, n5).
    4. Hitung EV, win rate, avg win/loss per fold.
    5. Agregasi semua fold + breakdown per score bucket.

    CATATAN PENTING tentang look-ahead bias:
    - Scoring dilakukan hanya menggunakan data sampai hari sinyal (tidak ada future data).
    - Exit price dihitung dari data yang sudah ada (sudah terjadi) — ini bukan look-ahead,
      karena kita hanya mengukur apa yang terjadi setelah sinyal.
    - Indikator dihitung dari rolling window, bukan dari seluruh dataset.

    Args:
        db: DatabaseManager instance
        engine: ScoringEngine yang sudah dikonfigurasi
        n_folds: Jumlah fold (default 5)
        min_lookback: Minimal hari historis sebelum mulai scoring (untuk indikator)
        min_score_to_signal: Minimum skor untuk dianggap sinyal (default dari config)
        target_threshold_pct: Target return persen untuk 'hit' (default dari config)
    """

    def __init__(
        self,
        db: DatabaseManager,
        engine: ScoringEngine = None,
        n_folds: int = 5,
        min_lookback: int = 60,
        min_score_to_signal: int = None,
        target_threshold_pct: float = None,
        max_date: str = None,
    ) -> None:
        self.db = db
        self.engine = engine or ScoringEngine()
        self.n_folds = n_folds
        self.min_lookback = min_lookback
        self.min_score_to_signal = min_score_to_signal or SCORING_CONFIG['bullish_threshold']
        self.target_threshold_pct = target_threshold_pct or SCORING_CONFIG['signal_return_threshold']
        self.max_date = max_date

    def run(self, tickers: list[str]) -> BacktestResult:
        """
        Jalankan walk-forward backtest untuk semua ticker.

        Args:
            tickers: List kode saham yang ada di database

        Returns:
            BacktestResult lengkap dengan semua fold + score bucket breakdown.
        """
        print(f"\n{'='*65}")
        print(f"  IDX-Screener v2 — Walk-Forward Backtest")
        print(f"  Tickers  : {len(tickers)}")
        print(f"  Folds    : {self.n_folds}")
        print(f"  Min score: {self.min_score_to_signal}")
        print(f"  Target   : {self.target_threshold_pct}% (gross) | "
              f"{self.target_threshold_pct + IDX_ROUNDTRIP_COST*100:.1f}% (net of fees)")
        print(f"{'='*65}")

        # Tentukan range tanggal dari semua data yang ada
        all_dates = self._get_all_dates(tickers)
        if len(all_dates) < self.min_lookback + 30:
            raise ValueError(f"Data tidak cukup: {len(all_dates)} hari, "
                             f"butuh minimal {self.min_lookback + 30}")

        # Buat fold windows (test windows saja — train adalah semua data sebelumnya)
        fold_windows = self._make_fold_windows(all_dates)

        result = BacktestResult(
            tickers=tickers,
            n_folds=self.n_folds,
            target_threshold_pct=self.target_threshold_pct,
        )

        for fold_idx, (test_start, test_end) in enumerate(fold_windows, 1):
            print(f"\n  Fold {fold_idx}/{self.n_folds}: "
                  f"Test [{test_start} → {test_end}]")

            fold_result = self._run_fold(
                fold_idx=fold_idx,
                tickers=tickers,
                all_dates=all_dates,
                test_start=test_start,
                test_end=test_end,
            )
            result.folds.append(fold_result)
            print(f"    Sinyal BULLISH: {len(fold_result.bullish_signals)}")

            ev = fold_result.expected_value()
            wr = fold_result.win_rate()
            if ev is not None:
                print(f"    EV (net fees) : {ev*100:+.2f}%")
                print(f"    Win Rate      : {wr*100:.1f}%" if wr else "    Win Rate: N/A")
            else:
                n = len(fold_result.bullish_signals)
                print(f"    EV: N/A (sample terlalu kecil: {n} sinyal bullish, butuh >= 10)")

        # Hitung score bucket breakdown
        result.score_buckets = self._compute_score_buckets(result)

        print(f"\n{'='*65}")
        print(f"  SELESAI")

        return result

    def _run_fold(
        self,
        fold_idx: int,
        tickers: list[str],
        all_dates: list[str],
        test_start: str,
        test_end: str,
    ) -> FoldResult:
        """Jalankan satu fold: score setiap hari di window test, catat outcome."""
        # Train start = awal data (minus lookback untuk warm-up indikator)
        # Kita load semua data dari awal untuk konteks indikator
        data_start = all_dates[0]

        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=data_start,
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
        )

        # Get test dates
        test_dates = [d for d in all_dates if test_start <= d <= test_end]

        for ticker in tickers:
            # Load seluruh data sampai akhir test window (termasuk beberapa hari extra
            # untuk hitung exit price n5)
            extra_end = self._offset_date(test_end, 10)  # +10 hari kalender
            df_full = self.db.get_prices_with_indicators(ticker, end=extra_end)

            if df_full.empty or len(df_full) < self.min_lookback:
                continue

            # Konversi index ke string untuk comparison
            df_full.index = pd.to_datetime(df_full.index)

            # Score setiap hari di window test
            for test_date in test_dates:
                try:
                    signal = self._score_one_day(ticker, df_full, test_date)
                    if signal:
                        fold.signals.append(signal)
                except Exception as exc:
                    # Jangan stop backtest karena satu error
                    pass

        return fold

    def _score_one_day(
        self,
        ticker: str,
        df_full: pd.DataFrame,
        target_date: str,
    ) -> Optional[SignalRecord]:
        """
        Score satu ticker pada satu hari. Return SignalRecord atau None.

        Ini adalah inti dari anti-look-ahead-bias:
        kita hanya pakai data sampai target_date saat scoring.
        """
        target_dt = pd.Timestamp(target_date)

        # Data yang tersedia sampai hari ini (inclusive) — simulasi "hari ini"
        df_until_today = df_full[df_full.index <= target_dt]
        if len(df_until_today) < self.min_lookback:
            return None

        # Score dengan konteks data historis (bukan masa depan)
        score_result = self.engine.score(ticker, df_until_today, today=target_date)
        score = score_result['skor_total']
        signal_label = score_result['sinyal']

        # Hanya catat sinyal di atas threshold
        if score < self.min_score_to_signal:
            return None

        # Entry price = close pada target_date
        row_today = df_until_today[df_until_today.index == target_dt]
        if row_today.empty:
            return None
        entry_price = float(row_today['Close'].iloc[0])

        # Exit prices: data SETELAH target_date (sudah terjadi, bukan prediction)
        df_after = df_full[df_full.index > target_dt].sort_index()

        def get_exit(n_trading_days: int) -> Optional[float]:
            if len(df_after) >= n_trading_days:
                return float(df_after['Close'].iloc[n_trading_days - 1])
            return None

        exit_n1 = get_exit(1)
        exit_n3 = get_exit(3)
        exit_n5 = get_exit(5)

        def calc_return(exit_price: Optional[float]) -> Optional[float]:
            if exit_price and entry_price > 0:
                return (exit_price - entry_price) / entry_price
            return None

        ret_n1 = calc_return(exit_n1)
        ret_n3 = calc_return(exit_n3)
        ret_n5 = calc_return(exit_n5)
        threshold = self.target_threshold_pct / 100

        return SignalRecord(
            ticker=ticker,
            date=target_date,
            score=score,
            signal=signal_label,
            entry_price=entry_price,
            exit_price_n1=exit_n1,
            exit_price_n3=exit_n3,
            exit_price_n5=exit_n5,
            return_n1=ret_n1,
            return_n3=ret_n3,
            return_n5=ret_n5,
            hit_target=ret_n3 >= threshold if ret_n3 is not None else None,
            hit_target_net=ret_n3 >= (threshold + IDX_ROUNDTRIP_COST) if ret_n3 is not None else None,
        )

    def _compute_score_buckets(self, result: BacktestResult) -> dict:
        """
        Hitung metrics per score bucket.

        KRITIS: Selalu laporkan sample count per bucket.
        Precision dari sample < 30 tidak meaningful.
        """
        all_signals = [s for fold in result.folds for s in fold.bullish_signals
                       if s.return_n3 is not None]

        buckets = {
            '50-64': [s for s in all_signals if 50 <= s.score <= 64],
            '65-74': [s for s in all_signals if 65 <= s.score <= 74],
            '75-84': [s for s in all_signals if 75 <= s.score <= 84],
            '85-100': [s for s in all_signals if 85 <= s.score <= 100],
        }

        threshold = self.target_threshold_pct / 100
        bucket_stats = {}
        for label, signals in buckets.items():
            n = len(signals)
            if n == 0:
                bucket_stats[label] = {'n': 0, 'note': 'Tidak ada sinyal'}
                continue

            returns = [s.return_n3 for s in signals]
            wins = [r for r in returns if r >= threshold]
            losses = [r for r in returns if r < threshold]
            wr = len(wins) / n
            avg_w = float(np.mean(wins)) if wins else 0
            avg_l = float(np.mean(losses)) if losses else 0
            ev_net = (wr * avg_w + (1-wr) * avg_l) - IDX_ROUNDTRIP_COST

            # Peringatan sample kecil
            reliability = 'OK' if n >= 30 else ('HATI-HATI' if n >= 15 else 'TIDAK VALID (sample terlalu kecil)')

            bucket_stats[label] = {
                'n': n,
                'win_rate': round(wr, 4),
                'avg_win': round(avg_w, 4),
                'avg_loss': round(avg_l, 4),
                'ev_net': round(ev_net, 4),
                'reliability': reliability,
            }

        return bucket_stats

    def _get_all_dates(self, tickers: list[str]) -> list[str]:
        """Ambil semua tanggal trading yang ada di database untuk tickers ini."""
        with self.db._connect() as conn:
            placeholders = ','.join('?' * len(tickers))
            query = f"SELECT DISTINCT date FROM daily_prices WHERE ticker IN ({placeholders}) AND is_valid=1"
            params = tickers
            if self.max_date:
                query += " AND date <= ?"
                params = tickers + [self.max_date]
            query += " ORDER BY date ASC"
            
            rows = conn.execute(query, params).fetchall()
        return [r[0] for r in rows]

    def _make_fold_windows(self, all_dates: list[str]) -> list[tuple[str, str]]:
        """
        Bagi data menjadi N fold walk-forward windows.
        Setiap fold punya test window berbeda, non-overlapping.
        Train window = semua data sebelum test window.
        """
        # Exclude min_lookback hari pertama (untuk warm-up indikator)
        usable_dates = all_dates[self.min_lookback:]

        if len(usable_dates) == 0:
            raise ValueError("Data tidak cukup setelah lookback period")

        fold_size = len(usable_dates) // self.n_folds
        windows = []
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx   = start_idx + fold_size - 1
            if i == self.n_folds - 1:
                end_idx = len(usable_dates) - 1  # fold terakhir ambil sisa semua
            windows.append((usable_dates[start_idx], usable_dates[end_idx]))

        return windows

    @staticmethod
    def _offset_date(date_str: str, days: int) -> str:
        """Tambah N hari ke date string."""
        dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=days)
        return dt.strftime('%Y-%m-%d')


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IDX-Screener v2 — Walk-Forward Backtester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python -m src.backtest.engine --tickers BBCA ASII TLKM --folds 5
  python -m src.backtest.engine --all-db --folds 5
  python -m src.backtest.engine --all-db --report backtest_report.html
  python -m src.backtest.engine --all-db --min-score 60 --folds 5
        """
    )
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='List ticker. Default: semua yang ada di DB.')
    parser.add_argument('--all-db', action='store_true',
                        help='Pakai semua ticker yang ada di database.')
    parser.add_argument('--folds', type=int, default=5,
                        help='Jumlah walk-forward fold (default: 5).')
    parser.add_argument('--min-score', type=int, default=None,
                        help='Minimum skor untuk dianggap sinyal (default: dari config).')
    parser.add_argument('--report', type=str, default=None,
                        help='Output report file path (HTML atau CSV).')
    parser.add_argument('--db', type=str, default=None,
                        help='Path ke SQLite database.')

    args = parser.parse_args()

    db_path = args.db or os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    engine = ScoringEngine()
    backtester = WalkForwardBacktester(
        db=db,
        engine=engine,
        n_folds=args.folds,
        min_score_to_signal=args.min_score,
    )

    if args.all_db or not args.tickers:
        tickers = db.get_tickers()
    else:
        tickers = [t.upper() + ('.JK' if not t.upper().endswith('.JK') else '')
                   for t in args.tickers]

    if not tickers:
        print("Error: Tidak ada ticker di database. Jalankan ingestion dulu.")
        print("  python -m src.data.ingestion")
        sys.exit(1)

    result = backtester.run(tickers)

    # Import report module di sini (bukan di top level, biar tidak circular)
    from backtest.report import BacktestReporter
    reporter = BacktestReporter(result)
    reporter.print_terminal()

    if args.report:
        if args.report.endswith('.html'):
            reporter.save_html(args.report)
        else:
            reporter.save_csv(args.report)


if __name__ == '__main__':
    main()
