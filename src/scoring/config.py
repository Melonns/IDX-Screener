"""
config.py — Scoring Configuration (v1 Rule-Based — LEGACY)

CATATAN:
    File ini berisi config untuk scoring engine v1 (rule-based, skor 0-100).
    Scanner v3 (observation_tags.py) TIDAK menggunakan config ini — v3 menggunakan
    relative percentile terhadap histori 60 hari masing-masing saham.

    Config ini dipertahankan untuk:
    1. Backward compatibility dengan backtest lama yang masih referensi scoring_version='rule_v1.0'
    2. Referensi dokumentasi threshold yang pernah diuji

    JANGAN tambah threshold baru di sini untuk scanner v3.
    Scanner v3 sudah menggunakan threshold relatif di observation_tags.py.

Soal target_threshold:
    Threshold minimum harus mempertimbangkan biaya transaksi IDX:
    - Fee beli: ~0.1-0.15% (tergantung sekuritas)
    - Fee jual: ~0.2-0.25% (termasuk PPh)
    - Total roundtrip: ~0.3-0.5%
    Jadi kalau target return 1%, net setelah fee cuma ~0.6%.
    Sebaiknya target minimal 1.5-2% untuk punya margin yang bermakna.
"""

SCORING_CONFIG: dict = {
    # ── LEGACY v1 Rule-Based Config ────────────────────────────────────────────
    # Config ini untuk backtest scoring v1 LAMA. Scanner v3 tidak pakai.

    # RSI thresholds (v1)
    'rsi_bullish_threshold': 50,
    'rsi_oversold_threshold': 35,
    'rsi_overbought_threshold': 75,

    # MACD thresholds (v1)
    'macd_cross_window': 3,

    # Volume thresholds (v1)
    'rvol_spike_threshold': 2.0,
    'rvol_moderate_threshold': 1.5,

    # Support/Resistance lookback (v1)
    'sr_lookback_short': 20,
    'sr_lookback_long': 50,

    # Signal classification thresholds (v1)
    'bullish_threshold': 65,
    'bearish_threshold': 35,

    # Backtest target return threshold
    'signal_return_threshold': 2.0,  # % return n+3 hari

    # Bollinger Band squeeze detection (v1)
    'bb_squeeze_percentile': 0.2,
    'bb_squeeze_window': 50,

    # Scoring version
    'scoring_version': 'rule_v1.0',
}

# ─── Scanner v3 Config ────────────────────────────────────────────────────────
# Threshold relatif untuk observation tags (sudah hardcode di observation_tags.py).
# Config ini hanya untuk dokumentasi dan override potensial di masa depan.

SCANNER_V3_CONFIG: dict = {
    # Observation tag thresholds (relative percentile)
    'volume_spike_percentile': 90,       # Volume ratio >= p90 dari 60 hari
    'rsi_extreme_high_percentile': 90,   # RSI >= p90 dari 60 hari
    'rsi_extreme_low_percentile': 10,    # RSI <= p10 dari 60 hari
    'bb_squeeze_percentile': 10,         # BB width <= p10 dari 60 hari
    'macd_cross_freshness_days': 3,      # Max hari sejak MACD cross
    'ema50_distance_pct': 5.0,           # % deviasi dari EMA50
    'atr_expansion_ratio': 1.5,          # ATR / avg_ATR_20 >= ini
    'gap_threshold_pct': 1.0,            # Gap up/down minimum %

    # Scan configuration
    'lookback_days': 60,                 # Hari histori untuk evaluation
    'rarity_lookback_days': 252,         # 12 bulan untuk rarity context
    'max_results': 15,                   # Max saham unusual ditampilkan
    'max_new_fetches_per_run': 25,       # Safety cap yfinance fetch per scan
}
