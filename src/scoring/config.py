"""
config.py — Semua threshold dan bobot scoring sebagai parameter.

PENTING: Semua angka di sini bisa dan harus diubah saat proses tuning.
Jangan hardcode threshold di dalam logic scoring engine.

Workflow tuning:
1. Ubah nilai di sini
2. Jalankan backtest
3. Bandingkan EV per sinyal dengan scoring_version berbeda
4. Pilih config yang punya EV dan win rate terbaik

Note soal target_threshold:
    Threshold minimum harus mempertimbangkan biaya transaksi IDX:
    - Fee beli: ~0.1-0.15% (tergantung sekuritas)
    - Fee jual: ~0.2-0.25% (termasuk PPh)
    - Total roundtrip: ~0.3-0.5%
    Jadi kalau target return 1%, net setelah fee cuma ~0.6%.
    Sebaiknya target minimal 1.5-2% untuk punya margin yang bermakna.
"""

SCORING_CONFIG: dict = {
    # ── RSI thresholds ─────────────────────────────────────────────────────
    'rsi_bullish_threshold': 50,        # RSI di atas ini + naik = momentum bullish
    'rsi_oversold_threshold': 35,       # (masih disimpen buat konteks)
    'rsi_overbought_threshold': 75,     # Dinaikkan ke 75 agar tidak memotong momentum

    # ── MACD thresholds ────────────────────────────────────────────────────
    'macd_cross_window': 3,             # max hari sejak cross 0 untuk dianggap fresh momentum

    # ── Volume thresholds ──────────────────────────────────────────────────
    'rvol_spike_threshold': 2.0,        # RVOL ≥ ini → volume spike (full score)
    'rvol_moderate_threshold': 1.5,     # RVOL ≥ ini → moderate (partial score)

    # ── Support/Resistance lookback ────────────────────────────────────────
    'sr_lookback_short': 20,            # hari untuk swing high-low jangka pendek
    'sr_lookback_long': 50,             # hari untuk swing high-low jangka menengah

    # ── Signal classification thresholds ──────────────────────────────────
    'bullish_threshold': 65,            # skor ≥ ini → label BULLISH
    'bearish_threshold': 35,            # skor ≤ ini → label BEARISH
                                        # antara keduanya → NEUTRAL

    # ── Backtest target return threshold ──────────────────────────────────
    # Mempertimbangkan biaya transaksi IDX (~0.4% roundtrip):
    # Target 2% berarti net setelah fee ≈ 1.6% — ini baru bermakna.
    # JANGAN set di bawah 1% karena setelah fee bisa breakeven.
    'signal_return_threshold': 2.0,     # % return n+3 hari yang dianggap "hit target"

    # ── Bollinger Band squeeze detection ──────────────────────────────────
    'bb_squeeze_percentile': 0.2,       # width di bawah percentile ini = squeeze
    'bb_squeeze_window': 50,            # window untuk kalkulasi percentile

    # ── Scoring version ────────────────────────────────────────────────────
    # Update ini setiap kali ada perubahan rule/threshold yang signifikan.
    # Penting untuk membandingkan performa antar versi di database.
    'scoring_version': 'rule_v1.0',
}
