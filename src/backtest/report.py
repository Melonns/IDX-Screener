"""
report.py — Backtest report generator untuk IDX-Screener v2.

Generate laporan backtest dalam format:
- Terminal output (default, selalu tampil)
- CSV export (--report file.csv)
- HTML report (--report file.html)

PENTING: Laporan ini selalu tampilkan sample count per bucket.
Lihat reliability label: OK / HATI-HATI / TIDAK VALID.
Jangan ambil keputusan dari bucket dengan label TIDAK VALID.
"""

import csv
import html
import os
from datetime import datetime
from pathlib import Path

from backtest.engine import BacktestResult, FoldResult, IDX_ROUNDTRIP_COST
from scoring.config import SCORING_CONFIG


class BacktestReporter:
    """
    Generate backtest report dalam berbagai format.

    Args:
        result: BacktestResult dari WalkForwardBacktester.run()
    """

    def __init__(self, result: BacktestResult) -> None:
        self.result = result

    # ─────────────────────────────────────────────────────────────────────────
    # Terminal Output
    # ─────────────────────────────────────────────────────────────────────────

    def print_terminal(self) -> None:
        """Print laporan lengkap ke terminal."""
        r = self.result
        metrics = r.aggregate_metrics()

        print(f"\n{'='*65}")
        print(f"  BACKTEST REPORT — IDX-Screener v2")
        print(f"  Tanggal   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Tickers   : {len(r.tickers)} saham")
        print(f"  Folds     : {r.n_folds}")
        print(f"  Target    : {r.target_threshold_pct}% gross / "
              f"{r.target_threshold_pct + IDX_ROUNDTRIP_COST*100:.1f}% net of fees")
        print(f"{'='*65}")

        if 'error' in metrics:
            print(f"\n  [!] {metrics['error']}")
            return

        # ── Aggregate Metrics ─────────────────────────────────────────────────
        print(f"\n  AGGREGATE (semua fold, sinyal BULLISH saja):")
        print(f"  {'─'*55}")
        print(f"  Total sinyal BULLISH     : {metrics['n_signals_bullish']:,}")
        print(f"  Fold dengan data cukup   : {metrics['n_folds_with_data']}/{r.n_folds}")
        print(f"  Win Rate (n+3)           : {metrics['win_rate']*100:.1f}%")
        print(f"  Avg Win                  : {metrics['avg_win']*100:+.2f}%")
        print(f"  Avg Loss                 : {metrics['avg_loss']*100:+.2f}%")
        print(f"  Win/Loss Ratio           : {metrics['win_loss_ratio']:.2f}x"
              if metrics['win_loss_ratio'] else "  Win/Loss Ratio           : N/A")
        print(f"  EV Gross                 : {metrics['ev_gross']*100:+.2f}%")
        print(f"  EV Net of Fees           : {metrics['ev_net_of_fees']*100:+.2f}%  "
              f"(setelah {IDX_ROUNDTRIP_COST*100:.1f}% roundtrip cost)")
        print(f"  Max Drawdown (n+3)       : {metrics['max_drawdown']*100:+.2f}%"
              if metrics['max_drawdown'] else "  Max Drawdown             : N/A")

        # ── Fold Consistency ──────────────────────────────────────────────────
        print(f"\n  KONSISTENSI ANTAR FOLD (kunci deteksi overfitting):")
        print(f"  {'─'*55}")
        for i, (ev, wr) in enumerate(zip(metrics['ev_per_fold'], metrics['wr_per_fold']), 1):
            ev_str = f"{ev*100:+.2f}%" if ev is not None else "N/A (sample kecil)"
            wr_str = f"{wr*100:.1f}%"  if wr is not None else "N/A"
            print(f"  Fold {i}:  EV={ev_str:<12}  WR={wr_str}")

        print(f"\n  EV Std antar fold        : "
              f"{metrics['ev_std_antar_fold']*100:.2f}%"
              if metrics['ev_std_antar_fold'] else "  EV Std antar fold        : N/A")
        print(f"  EV Sharpe-like           : "
              f"{metrics['ev_sharpe_like']:.3f}"
              if metrics['ev_sharpe_like'] else "  EV Sharpe-like           : N/A")
        print(f"  Fold dengan EV positif   : {metrics['n_fold_ev_positif']}/{r.n_folds}")

        # ── Score Bucket Breakdown ────────────────────────────────────────────
        print(f"\n  BREAKDOWN PER SCORE BUCKET:")
        print(f"  {'─'*55}")
        print(f"  {'Bucket':<10} {'N':>5} {'WR':>7} {'AvgWin':>8} {'AvgLoss':>9} "
              f"{'EV(net)':>9}  Reliability")
        print(f"  {'─'*55}")
        for bucket, stats in self.result.score_buckets.items():
            if stats['n'] == 0:
                print(f"  {bucket:<10} {'0':>5}  (tidak ada sinyal)")
                continue
            print(
                f"  {bucket:<10} "
                f"{stats['n']:>5} "
                f"{stats['win_rate']*100:>6.1f}% "
                f"{stats['avg_win']*100:>+7.2f}% "
                f"{stats['avg_loss']*100:>+8.2f}% "
                f"{stats['ev_net']*100:>+8.2f}%  "
                f"{stats['reliability']}"
            )

        # ── Gate Check ────────────────────────────────────────────────────────
        gate = metrics['lolos_gate']
        print(f"\n  PHASE 1 GATE CHECK:")
        print(f"  {'─'*55}")
        self._print_gate_item("EV net > 0.3%", gate['ev_net_ok'], metrics['ev_net_of_fees'])
        self._print_gate_item("Win rate >= 55%", gate['wr_ok'], metrics['win_rate'])
        self._print_gate_item("Avg Win / Avg Loss >= 1.2x",
                               gate['wl_ratio_ok'], metrics['win_loss_ratio'])
        self._print_gate_item(f"EV positif >= 4/{r.n_folds} fold",
                               gate['fold_consistency'], metrics['n_fold_ev_positif'])

        print(f"\n  {'─'*55}")
        if gate['semua_lolos']:
            print(f"  [OK] SEMUA KRITERIA TERPENUHI.")
            print(f"       Sistem siap dilanjutkan ke Telegram / Phase 2.")
        else:
            gagal = [k for k, v in gate.items() if not v and k != 'semua_lolos']
            print(f"  [!!] BELUM LOLOS GATE. Gagal di: {', '.join(gagal)}")
            print(f"       Jangan lanjutkan ke Telegram atau Phase 2 dulu.")
            print(f"       Pertimbangkan: tuning threshold di scoring/config.py,")
            print(f"       atau cek apakah indikator perlu penyesuaian.")

        print(f"  {'─'*55}")
        print(f"  [!] Ingat: backtest ini menggunakan data historis.")
        print(f"      Hasil masa lalu tidak menjamin performa masa depan.")
        print(f"      Paper trading 2-4 minggu sebelum live notifikasi.")
        print(f"{'='*65}\n")

    def _print_gate_item(self, label: str, passed: bool, value) -> None:
        icon   = "[OK]" if passed else "[XX]"
        val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {icon} {label:<35} (nilai: {val_str})")

    # ─────────────────────────────────────────────────────────────────────────
    # CSV Export
    # ─────────────────────────────────────────────────────────────────────────

    def save_csv(self, filepath: str) -> None:
        """Export semua sinyal ke CSV untuk analisis lanjutan di Excel."""
        all_signals = [s for fold in self.result.folds for s in fold.bullish_signals]

        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'fold', 'ticker', 'date', 'score', 'signal',
                'entry_price', 'exit_n1', 'exit_n3', 'exit_n5',
                'return_n1_pct', 'return_n3_pct', 'return_n5_pct',
                'hit_target', 'hit_target_net',
            ])
            writer.writeheader()
            for fold in self.result.folds:
                for s in fold.bullish_signals:
                    writer.writerow({
                        'fold': fold.fold_idx,
                        'ticker': s.ticker,
                        'date': s.date,
                        'score': s.score,
                        'signal': s.signal,
                        'entry_price': s.entry_price,
                        'exit_n1': s.exit_price_n1,
                        'exit_n3': s.exit_price_n3,
                        'exit_n5': s.exit_price_n5,
                        'return_n1_pct': f"{s.return_n1*100:.2f}" if s.return_n1 else '',
                        'return_n3_pct': f"{s.return_n3*100:.2f}" if s.return_n3 else '',
                        'return_n5_pct': f"{s.return_n5*100:.2f}" if s.return_n5 else '',
                        'hit_target': s.hit_target,
                        'hit_target_net': s.hit_target_net,
                    })

        print(f"CSV tersimpan: {filepath} ({len(all_signals)} sinyal)")

    # ─────────────────────────────────────────────────────────────────────────
    # HTML Report
    # ─────────────────────────────────────────────────────────────────────────

    def save_html(self, filepath: str) -> None:
        """Generate HTML report yang bisa dibuka di browser."""
        metrics = self.result.aggregate_metrics()
        gate    = metrics.get('lolos_gate', {})

        gate_color = '#22c55e' if gate.get('semua_lolos') else '#ef4444'
        gate_text  = 'LOLOS ✓' if gate.get('semua_lolos') else 'BELUM LOLOS ✗'

        def pct(v):
            return f"{v*100:+.2f}%" if v is not None else "N/A"

        def num(v, dec=2):
            return f"{v:.{dec}f}" if v is not None else "N/A"

        bucket_rows = ""
        for bucket, stats in self.result.score_buckets.items():
            if stats['n'] == 0:
                bucket_rows += f"<tr><td>{bucket}</td><td>0</td>" + "<td>-</td>"*5 + "<td>-</td></tr>\n"
                continue
            rel_color = {'OK': '#22c55e', 'HATI-HATI': '#f59e0b', }.get(
                stats['reliability'].split()[0], '#ef4444')
            bucket_rows += f"""
            <tr>
                <td>{bucket}</td>
                <td>{stats['n']}</td>
                <td>{stats['win_rate']*100:.1f}%</td>
                <td>{stats['avg_win']*100:+.2f}%</td>
                <td>{stats['avg_loss']*100:+.2f}%</td>
                <td>{stats['ev_net']*100:+.2f}%</td>
                <td style="color:{rel_color};font-weight:bold">{stats['reliability']}</td>
            </tr>"""

        fold_rows = ""
        for i, (ev, wr) in enumerate(zip(metrics.get('ev_per_fold', []),
                                          metrics.get('wr_per_fold', [])), 1):
            fold_rows += f"""
            <tr>
                <td>Fold {i}</td>
                <td>{pct(ev)}</td>
                <td>{pct(wr)}</td>
            </tr>"""

        html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IDX-Screener v2 — Backtest Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: .3rem;
       background: linear-gradient(135deg, #38bdf8, #818cf8);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #94a3b8; margin: 1.5rem 0 .7rem; }}
  .meta {{ font-size: .85rem; color: #64748b; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; }}
  .card-label {{ font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }}
  .card-value {{ font-size: 1.6rem; font-weight: 700; margin-top: .3rem; }}
  .green {{ color: #22c55e; }}
  .red   {{ color: #ef4444; }}
  .yellow {{ color: #f59e0b; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b;
           border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
  th {{ background: #0f172a; padding: .8rem 1rem; text-align: left;
        font-size: .8rem; text-transform: uppercase; color: #64748b; }}
  td {{ padding: .7rem 1rem; border-top: 1px solid #1e293b; font-size: .9rem; }}
  tr:nth-child(even) td {{ background: #263347; }}
  .gate-box {{ background: #1e293b; border: 2px solid {gate_color}; border-radius: 12px;
               padding: 1.5rem; margin-top: 1.5rem; }}
  .gate-label {{ font-size: 1.2rem; font-weight: 700; color: {gate_color}; }}
  .warning {{ background: #1c1917; border: 1px solid #78350f; border-radius: 8px;
              padding: 1rem; margin-top: 1.5rem; font-size: .85rem; color: #fbbf24; }}
</style>
</head>
<body>
<h1>IDX-Screener v2 — Backtest Report</h1>
<p class="meta">
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
  {len(self.result.tickers)} tickers &nbsp;|&nbsp;
  {self.result.n_folds} folds &nbsp;|&nbsp;
  Target: {self.result.target_threshold_pct}% gross
</p>

<h2>Aggregate Metrics</h2>
<div class="grid">
  <div class="card">
    <div class="card-label">Total Sinyal Bullish</div>
    <div class="card-value">{metrics.get('n_signals_bullish', 0):,}</div>
  </div>
  <div class="card">
    <div class="card-label">Win Rate (n+3)</div>
    <div class="card-value {'green' if metrics.get('win_rate',0)>=0.55 else 'red'}">{pct(metrics.get('win_rate'))}</div>
  </div>
  <div class="card">
    <div class="card-label">EV Net of Fees</div>
    <div class="card-value {'green' if (metrics.get('ev_net_of_fees') or 0)>0 else 'red'}">{pct(metrics.get('ev_net_of_fees'))}</div>
  </div>
  <div class="card">
    <div class="card-label">Win/Loss Ratio</div>
    <div class="card-value {'green' if (metrics.get('win_loss_ratio') or 0)>=1.2 else 'red'}">{num(metrics.get('win_loss_ratio'))}x</div>
  </div>
  <div class="card">
    <div class="card-label">EV Sharpe-like</div>
    <div class="card-value">{num(metrics.get('ev_sharpe_like'), 3)}</div>
  </div>
  <div class="card">
    <div class="card-label">Fold EV Positif</div>
    <div class="card-value {'green' if metrics.get('n_fold_ev_positif',0)>=4 else 'red'}">{metrics.get('n_fold_ev_positif', 0)}/{self.result.n_folds}</div>
  </div>
</div>

<h2>Konsistensi Antar Fold</h2>
<table>
  <tr><th>Fold</th><th>EV Net</th><th>Win Rate</th></tr>
  {fold_rows}
</table>

<h2>Breakdown per Score Bucket</h2>
<p style="font-size:.8rem;color:#64748b;margin-bottom:.5rem">
  Perhatikan kolom N (sample count). Bucket dengan sample &lt; 30 tidak reliable.
</p>
<table>
  <tr>
    <th>Bucket</th><th>N Sinyal</th><th>Win Rate</th>
    <th>Avg Win</th><th>Avg Loss</th><th>EV Net</th><th>Reliability</th>
  </tr>
  {bucket_rows}
</table>

<div class="gate-box">
  <div class="gate-label">Phase 1 Gate: {gate_text}</div>
  <ul style="margin-top:.8rem;padding-left:1.2rem;font-size:.9rem;color:#94a3b8">
    <li>EV net &gt; 0.3%: {'✓' if gate.get('ev_net_ok') else '✗'} ({pct(metrics.get('ev_net_of_fees'))})</li>
    <li>Win rate &gt;= 55%: {'✓' if gate.get('wr_ok') else '✗'} ({pct(metrics.get('win_rate'))})</li>
    <li>Avg Win/Loss &gt;= 1.2x: {'✓' if gate.get('wl_ratio_ok') else '✗'} ({num(metrics.get('win_loss_ratio'))}x)</li>
    <li>EV positif &gt;= 4 fold: {'✓' if gate.get('fold_consistency') else '✗'} ({metrics.get('n_fold_ev_positif', 0)}/{self.result.n_folds} fold)</li>
  </ul>
</div>

<div class="warning">
  ⚠️ Hasil backtest menggunakan data historis. Survivorship bias mungkin ada
  (saham yang delisting tidak ikut training). Performa masa lalu tidak menjamin
  hasil ke depan. Lakukan paper trading 2–4 minggu sebelum live.
</div>

</body>
</html>"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report tersimpan: {filepath}")
