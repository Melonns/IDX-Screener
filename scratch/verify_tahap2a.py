"""
Dua pengecekan sebelum Tahap 2b:
1. Konfirmasi vol_accum_5d beneran dipake (bukan bug copy-paste)
2. Spearman correlation antara rel_strength_5d_rank vs vol_accum_5d_rank
"""
import sys, os
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from data.database import DatabaseManager
import config as app_config

db = DatabaseManager(os.path.join(app_config.DATA_DIR, 'idx_screener.db'))

print("=== VERIFIKASI 1: vol_accum_5d beneran dipake ===")
# Cek langsung di DB: ambil sample data dan tunjukkan nilainya berbeda dari rel_strength
with db._connect() as conn:
    df = pd.read_sql_query("""
        SELECT date, ticker, rel_strength_5d, rel_strength_5d_rank,
               vol_accum_5d, vol_accum_5d_rank, turnover_5d
        FROM contextual_indicators
        WHERE date = (SELECT MAX(date) FROM contextual_indicators WHERE date <= '2026-02-09')
        ORDER BY rel_strength_5d_rank ASC
        LIMIT 10
    """, conn)

print("Sample data (saham dengan rel_strength rank terendah):")
print(df[['ticker', 'rel_strength_5d_rank', 'vol_accum_5d_rank']].to_string())

print("\nApakah rel_strength_5d_rank == vol_accum_5d_rank?")
identical = (df['rel_strength_5d_rank'].round(2) == df['vol_accum_5d_rank'].round(2)).all()
print(f"  Identik? {'YA — PERLU INVESTIGASI' if identical else 'TIDAK — Fitur berbeda, aman'}")

# Juga cek korelasi Spearman antar dua rank
print("\n=== VERIFIKASI 2: Spearman Correlation antara dua rank ===")
with db._connect() as conn:
    full_df = pd.read_sql_query("""
        SELECT date, ticker, rel_strength_5d_rank, vol_accum_5d_rank
        FROM contextual_indicators
        WHERE date <= '2026-02-09'
        AND rel_strength_5d_rank IS NOT NULL
        AND vol_accum_5d_rank IS NOT NULL
    """, conn)

corr, pval = spearmanr(full_df['rel_strength_5d_rank'], full_df['vol_accum_5d_rank'])
print(f"N observasi      : {len(full_df)}")
print(f"Spearman r       : {corr:+.4f}")
print(f"P-value          : {pval:.4e}")

if abs(corr) < 0.3:
    verdict = "RENDAH (<0.3) — Dua fitur nangkep informasi berbeda. ADDITIVE."
elif abs(corr) < 0.6:
    verdict = "MODERAT (0.3-0.6) — Ada overlap tapi masih bisa saling melengkapi."
else:
    verdict = "TINGGI (>0.6) — Banyak overlap. Kombinasi mungkin tidak additive."
print(f"Verdict          : {verdict}")

# Distribusi per hari (cross-sectional correlation harian)
daily_corrs = []
for date, group in full_df.groupby('date'):
    if len(group) >= 5:
        c, _ = spearmanr(group['rel_strength_5d_rank'], group['vol_accum_5d_rank'])
        if not np.isnan(c):
            daily_corrs.append(c)

print(f"\nCross-sectional correlation harian:")
print(f"  Mean  : {np.mean(daily_corrs):+.4f}")
print(f"  Median: {np.median(daily_corrs):+.4f}")
print(f"  Std   : {np.std(daily_corrs):.4f}")

print("\n=== VERIFIKASI 3: Kenapa params identik persis? ===")
print("TPESampler(seed=42) dengan search space yang sama → startup trials")
print("(biasanya 10-25 trial pertama) sampling random deterministic.")
print("Trial ke-2 di dua studi yang berbeda menghasilkan angka yang sama persis")
print("karena urutan random number generator-nya sama (seed sama, search space sama).")
print(f"\nBukti beda fitur yang dipake: objective value BERBEDA:")
print(f"  Tahap 1 (rel_strength) @ params sama: -0.004998")
print(f"  Tahap 2a (vol_accum)   @ params sama: -0.004092")
print(f"  Kalau bug (fitur sama): harusnya objective identik, bukan beda 0.000906")
print(f"  → Konfirmasi: dua fitur beneran berbeda dalam perhitungan ✓")
