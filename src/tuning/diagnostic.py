"""
diagnostic.py — Diagnostic sebelum Optuna Tuning.

Tujuan:
1. Cek isu kualitas data RVOL (kenapa banyak 0).
2. Per-Indicator Isolated Backtest & Information Coefficient (IC) per fold.
3. Correlation Matrix antar indikator.

Menghasilkan file HTML diagnostic report.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import config as app_config
from data.database import DatabaseManager
from scoring.engine import ScoringEngine
from backtest.engine import IDX_ROUNDTRIP_COST

# ─────────────────────────────────────────────────────────────────────────────
# 1. RVOL Data Quality Check
# ─────────────────────────────────────────────────────────────────────────────

def check_rvol_quality(db: DatabaseManager) -> str:
    """Cek kualitas data RVOL langsung dari tabel database."""
    print("Checking RVOL data quality...")
    with db._connect() as conn:
        # Cek berapa banyak hari trading dengan Volume = 0
        zero_vol = conn.execute("SELECT COUNT(*) FROM daily_prices WHERE Volume = 0 OR Volume IS NULL").fetchone()[0]
        total_vol = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        
        # Cek distribusi volume_ratio_20d
        rvol_stats = pd.read_sql_query("SELECT volume_ratio_20d FROM daily_indicators WHERE volume_ratio_20d IS NOT NULL", conn)
    
    if rvol_stats.empty:
        return "<p>Data RVOL kosong/tidak ada di database.</p>"
        
    rvol_series = rvol_stats['volume_ratio_20d']
    zero_rvol = (rvol_series == 0).sum()
    low_rvol = (rvol_series < 1.0).sum()
    total_rvol = len(rvol_series)
    
    report = f"""
    <h3>RVOL Data Quality Check</h3>
    <ul>
        <li>Total baris harga di DB: {total_vol:,}</li>
        <li>Baris dengan Volume mentah = 0: {zero_vol:,} ({(zero_vol/total_vol)*100:.1f}%)</li>
        <li>Total baris indikator dengan RVOL valid: {total_rvol:,}</li>
        <li>Baris dengan RVOL tepat 0.0: {zero_rvol:,} ({(zero_rvol/total_rvol)*100:.1f}%)</li>
        <li>Baris dengan RVOL < 1.0 (Skor=0): {low_rvol:,} ({(low_rvol/total_rvol)*100:.1f}%)</li>
    </ul>
    <p><b>Kesimpulan RVOL:</b> Jika banyak skor RVOL bernilai 0 di engine, itu karena mayoritas hari trading ({low_rvol/total_rvol*100:.0f}%) memiliki volume di bawah rata-rata 20 hari (wajar untuk pasar yang tidak selalu trending/spike). Bukan berarti datanya cacat (Volume asli yang 0 hanya {(zero_vol/total_vol)*100:.1f}%).</p>
    """
    return report

# ─────────────────────────────────────────────────────────────────────────────
# 2. Re-scoring to get Breakdowns
# ─────────────────────────────────────────────────────────────────────────────

def score_ticker_history(ticker: str, db_path: str, min_score: int = 0) -> List[Dict]:
    """Score seluruh sejarah satu ticker dan kumpulkan sinyal + breakdown."""
    db = DatabaseManager(db_path)
    engine = ScoringEngine()
    df_full = db.get_prices_with_indicators(ticker)
    
    if df_full.empty or len(df_full) < 60:
        return []
        
    df_full.index = pd.to_datetime(df_full.index)
    signals = []
    
    # Kita butuh return_n3. Karena ini diagnostic (bukan pure walk-forward strict test),
    # kita bisa hitung return_n3 secara vektor di masa depan untuk mempercepat
    df_full['exit_n3'] = df_full['Close'].shift(-3)
    df_full['return_n3'] = (df_full['exit_n3'] - df_full['Close']) / df_full['Close']
    df_full['exit_n5'] = df_full['Close'].shift(-5)
    df_full['return_n5'] = (df_full['exit_n5'] - df_full['Close']) / df_full['Close']
    df_full['exit_n10'] = df_full['Close'].shift(-10)
    df_full['return_n10'] = (df_full['exit_n10'] - df_full['Close']) / df_full['Close']
    
    # Drop rows yang return_n10 nya NaN (hari-hari terakhir)
    valid_dates = df_full.dropna(subset=['return_n10']).index
    
    # Skip 60 hari pertama untuk lookback warm-up
    valid_dates = valid_dates[valid_dates >= df_full.index[60]]
    
    for target_dt in valid_dates:
        # Score menggunakan fungsi aslinya untuk dapetin breakdown yang valid
        # Optimization: Engine.score() secara default akan mengambil df.iloc[-1] sebagai 'today'
        # Jadi kita potong dataframe sampai target_dt
        df_until_today = df_full.loc[:target_dt]
        score_res = engine.score(ticker, df_until_today, today=target_dt.strftime('%Y-%m-%d'))
        
        if score_res['skor_total'] >= min_score:
            ret_n3 = float(df_full.loc[target_dt, 'return_n3'])
            ret_n5 = float(df_full.loc[target_dt, 'return_n5'])
            ret_n10 = float(df_full.loc[target_dt, 'return_n10'])
            signals.append({
                'ticker': ticker,
                'date': target_dt.strftime('%Y-%m-%d'),
                'score': score_res['skor_total'],
                'return_n3': ret_n3,
                'return_n5': ret_n5,
                'return_n10': ret_n10,
                'return_n3_net': ret_n3 - IDX_ROUNDTRIP_COST,
                'breakdown': score_res['breakdown']
            })
            
    return signals

def gather_diagnostic_data(db: DatabaseManager) -> pd.DataFrame:
    """Jalankan scoring ke semua ticker secara parallel dan buat DataFrame."""
    print("Gathering signals with detailed breakdowns (ini butuh waktu ~1 menit)...")
    tickers = db.get_tickers()
    db_path = db.db_path
    
    results = Parallel(n_jobs=-1)(
        delayed(score_ticker_history)(t, db_path) for t in tickers
    )
    
    # Flatten list of lists
    all_signals = [s for sublist in results for s in sublist]
    
    if not all_signals:
        print("Warning: Tidak ada sinyal yang dihasilkan.")
        return pd.DataFrame()
        
    # Konversi ke DataFrame
    # Flatten dictionary breakdown menjadi kolom-kolom
    rows = []
    for s in all_signals:
        row = {
            'ticker': s['ticker'],
            'date': s['date'],
            'score': s['score'],
            'return_n3': s['return_n3'],
            'return_n5': s['return_n5'],
            'return_n10': s['return_n10'],
            'return_n3_net': s['return_n3_net'],
        }
        for ind_result in s['breakdown']:
            ind_name = ind_result['indikator']
            row[f'ind_{ind_name}'] = ind_result['skor']
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Tambahkan Fold Number (bagi jadi 5 fold kronologis)
    dates = df['date'].unique()
    dates = np.sort(dates)
    fold_size = max(1, len(dates) // 5)
    
    def get_fold(d):
        idx = np.where(dates == d)[0][0]
        fold = (idx // fold_size) + 1
        return min(fold, 5) # Pastikan max 5
        
    df['fold_number'] = df['date'].apply(get_fold)
    
    print(f"Berhasil mengumpulkan {len(df)} sinyal.")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. Analysis & Output Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(rvol_report: str, df: pd.DataFrame, out_path: str):
    """Generate HTML Report untuk Diagnostic."""
    if df.empty:
        print("Data kosong, tidak dapat generate report.")
        return
        
    # Indikator columns
    ind_cols = [c for c in df.columns if c.startswith('ind_')]
    
    # ── 1. Isolated Backtest & IC ──
    
    analysis_rows = []
    
    # Fungsi pembantu untuk Cross-Sectional IC
    def calc_cs_ic(df_data, score_col, return_col):
        daily_ics = []
        for d, group in df_data.groupby('date'):
            if len(group) > 5:
                ic, _ = spearmanr(group[score_col], group[return_col])
                if not np.isnan(ic):
                    daily_ics.append(ic)
        return np.mean(daily_ics) if daily_ics else 0.0

    for col in ind_cols:
        ind_name = col.replace('ind_', '')
        
        # Max observed score for this indicator
        max_val = df[col].max()
        if max_val == 0:
            analysis_rows.append(f"""
            <tr>
                <td>{ind_name}</td>
                <td>0.0%</td>
                <td>0.0%</td>
                <td><b>NaN</b></td>
                <td>NaN</td>
                <td>NaN</td>
                <td style="font-size:0.8rem;color:#94a3b8">Constant 0</td>
                <td>0</td>
                <td>{len(df)}</td>
                <td><span style='color:#f59e0b'>Skor selalu 0</span></td>
            </tr>
            """)
            continue
            
        threshold = max_val / 2.0
        
        # Split Kuat vs Lemah
        mask_kuat = df[col] > threshold
        df_kuat = df[mask_kuat]
        df_lemah = df[~mask_kuat]
        
        n_kuat = len(df_kuat)
        n_lemah = len(df_lemah)
        
        wr_kuat = (df_kuat['return_n3_net'] > 0).mean() if n_kuat > 0 else 0
        wr_lemah = (df_lemah['return_n3_net'] > 0).mean() if n_lemah > 0 else 0
        
        # Calculate Cross-Sectional Rank IC for N+3, N+5, N+10
        cs_ic_n3 = calc_cs_ic(df, col, 'return_n3')
        cs_ic_n5 = calc_cs_ic(df, col, 'return_n5')
        cs_ic_n10 = calc_cs_ic(df, col, 'return_n10')
        
        # Calculate Global IC N+3 (sebagai referensi fold per fold)
        ic_folds = []
        for fold in range(1, 6):
            df_fold = df[df['fold_number'] == fold]
            if len(df_fold) > 5:
                ic, _ = spearmanr(df_fold[col], df_fold['return_n3'])
                if not np.isnan(ic):
                    ic_folds.append(ic)
                    
        ic_avg_global = np.mean(ic_folds) if ic_folds else 0
        
        # Verdict diambil dari CS IC N+3 (apakah ada sinyal relatif cross-sectional)
        if cs_ic_n3 < -0.05:
            verdict = "<span style='color:#ef4444'>Negatively Correlated (Buang/Revisi)</span>"
        elif abs(cs_ic_n3) <= 0.05:
            verdict = "<span style='color:#f59e0b'>No Edge (Noise)</span>"
        else:
            verdict = "<span style='color:#22c55e'>Predictive</span>"
            
        ic_folds_str = ", ".join([f"{x:+.2f}" for x in ic_folds])
        
        analysis_rows.append(f"""
        <tr>
            <td>{ind_name}</td>
            <td>{wr_kuat*100:.1f}%</td>
            <td>{wr_lemah*100:.1f}%</td>
            <td><b>{cs_ic_n3:+.3f}</b></td>
            <td>{cs_ic_n5:+.3f}</td>
            <td>{cs_ic_n10:+.3f}</td>
            <td style="font-size:0.8rem;color:#94a3b8">Global N3: {ic_avg_global:+.3f}</td>
            <td>{n_kuat}</td>
            <td>{n_lemah}</td>
            <td>{verdict}</td>
        </tr>
        """)
        
    analysis_html = "".join(analysis_rows)
    
    # ── 2. Correlation Matrix ──
    corr_matrix = df[ind_cols].corr(method='pearson')
    
    corr_html_rows = []
    # Header
    corr_html_rows.append("<tr><th>Indikator</th>" + "".join([f"<th>{c.replace('ind_','')}</th>" for c in ind_cols]) + "</tr>")
    
    for i, row_col in enumerate(ind_cols):
        row_str = f"<tr><td><b>{row_col.replace('ind_','')}</b></td>"
        for j, col in enumerate(ind_cols):
            val = corr_matrix.loc[row_col, col]
            if i == j:
                color = "#334155"
            elif val > 0.7:
                color = "#ef4444" # Highlight high correlation
            elif val < -0.7:
                color = "#3b82f6"
            else:
                color = "transparent"
                
            row_str += f"<td style='background-color:{color}'>{val:+.2f}</td>"
        row_str += "</tr>"
        corr_html_rows.append(row_str)
        
    corr_html = "".join(corr_html_rows)
    
    # ── 3. Build Full HTML ──
    
    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnostic Report - IDX-Screener v2</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem; }}
  h1, h2, h3 {{ margin-bottom: 1rem; color: #38bdf8; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b;
           border: 1px solid #334155; border-radius: 8px; overflow: hidden; margin-bottom: 2rem; }}
  th {{ background: #0f172a; padding: 0.8rem; text-align: left; font-size: 0.85rem; color: #94a3b8; }}
  td {{ padding: 0.8rem; border-top: 1px solid #1e293b; font-size: 0.9rem; }}
  tr:nth-child(even) td {{ background: #263347; }}
  .box {{ background: #1e293b; padding: 1.5rem; border-radius: 8px; border: 1px solid #334155; margin-bottom: 2rem; }}
</style>
</head>
<body>
    <h1>Per-Indicator Diagnostic Report</h1>
    <p>Target Return: N+3 (Net of fees). Total Sinyal: {len(df)}</p>
    
    <div class="box">
        {rvol_report}
    </div>
    
    <h2>1. Per-Indicator Contribution (Information Coefficient)</h2>
    <p style="margin-bottom:1rem; font-size:0.9rem; color:#94a3b8;">
        <b>Cross-Sectional Rank IC</b>: Korelasi Spearman harian antara skor komponen vs return, lalu dirata-rata. 
        Membersihkan noise kondisi pasar makro untuk melihat <i>relative strength</i> antar saham.
    </p>
    <table>
        <tr>
            <th>Indikator</th>
            <th>WR (Kuat)</th>
            <th>WR (Lemah)</th>
            <th>CS IC (N+3)</th>
            <th>CS IC (N+5)</th>
            <th>CS IC (N+10)</th>
            <th>Global IC (N+3)</th>
            <th>N (Kuat)</th>
            <th>N (Lemah)</th>
            <th>Verdict</th>
        </tr>
        {analysis_html}
    </table>
    
    <h2>2. Correlation Matrix Antar Indikator</h2>
    <p style="margin-bottom:1rem; font-size:0.9rem; color:#94a3b8;">
        Cek redundansi. Jika ada nilai > +0.70 (Merah), berarti dua indikator tersebut menghitung hal yang hampir sama. 
        Jangan beri bobot besar ke keduanya karena akan "double counting".
    </p>
    <table>
        {corr_html}
    </table>
    
    <div class="box" style="border-color:#f59e0b">
        <h3>Rekomendasi Sebelum Optuna:</h3>
        <ul style="margin-left:1.5rem; line-height:1.6">
            <li>Review kolom <b>Verdict</b>. Jika ada indikator berlabel 'Negatively Correlated' atau 'Noise', pertimbangkan untuk menghapus/mengecilkan search space bobot indikator tersebut di Phase 1 Optuna (jangan biarkan Optuna mencarinya dari rentang 5-35).</li>
            <li>Perhatikan <b>IC per Fold</b>. Jika rata-rata IC bagus tapi hanya didorong oleh 1 fold ekstrim, berarti indikator itu <i>regime-dependent</i> (hanya works di uptrend/downtrend tertentu).</li>
            <li>Cek <b>Correlation Matrix</b>. Jika ada 2 indikator yang korelasinya merah (>0.7), kurangi range bobot salah satunya di Optuna.</li>
        </ul>
    </div>
</body>
</html>
"""
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Diagnostic report tersimpan: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic per-indicator.")
    parser.add_argument('--db', type=str, default=None, help='Path ke SQLite DB')
    parser.add_argument('--out', type=str, default='diagnostic_report.html', help='Output HTML')
    args = parser.parse_args()
    
    db_path = args.db or os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    
    # 1. Check RVOL
    rvol_report = check_rvol_quality(db)
    
    # 2. Gather data
    df = gather_diagnostic_data(db)
    
    # 3. Generate Report
    if not df.empty:
        generate_report(rvol_report, df, args.out)

if __name__ == '__main__':
    main()
