# Spec: Upgrade Bot Deteksi Saham IDX (v2 — Technical Analysis Based)

## Konteks & Tujuan
Sebelumnya ada bot Telegram yang deteksi ARA/bullish/bearish saham IDX pakai data dari Yahoo Finance, tapi hasilnya tidak reliable karena tidak ada dasar teknikal yang jelas — sinyal terasa seperti tebakan tanpa alasan yang bisa diaudit.

Tujuan v2: bangun ulang sistem dengan pendekatan **quantitative scoring berbasis indikator teknikal**, jalan **lokal dulu** (pakai SQLite), dengan arsitektur modular supaya sumber data bisa diganti nanti tanpa rombak total. Target akhir adalah alat bantu screening saham untuk trading real, jadi akurasi dan auditability sinyal jadi prioritas, bukan cuma fitur banyak.

---

## Prinsip Desain
1. **Modular data source** — buat interface/abstraction layer untuk data provider, supaya nanti gampang ganti dari yfinance ke provider lain (Stockbit, RTI, dsb) tanpa ubah logic scoring/backtest.
2. **Scoring, bukan sinyal biner** — output harus berupa skor (misal 0–100) dengan breakdown alasan per komponen, bukan cuma label "bullish/bearish".
3. **Semua sinyal harus bisa diaudit** — simpan alasan tiap skor (indikator apa yang trigger, nilainya berapa) supaya bisa dievaluasi manual dan otomatis nanti.
4. **Backtest dulu sebelum live** — sistem scoring tidak boleh dipakai untuk notifikasi real sebelum melewati proses backtest dan walk-forward validation.
5. **Local-first** — semua development dan backtest jalan di lokal pakai SQLite. Deployment ke VPS/server itu fase belakangan, setelah sistem scoring divalidasi.

---

## Struktur Sistem yang Dibutuhkan

### 1. Data Layer
- Ambil data historis OHLCV harian dari yfinance untuk saham `.JK` (contoh: `BBCA.JK`).
- Simpan ke SQLite, jangan cuma fetch real-time tiap kali butuh — ini penting untuk backtest dan biar tidak bolak-balik hit API.
- Buat schema tabel minimal:
  - `stocks` (kode saham, nama, sektor, market cap kalau tersedia)
  - `daily_prices` (kode saham, tanggal, open, high, low, close, volume)
  - `signals` (kode saham, tanggal, skor, breakdown alasan dalam JSON, versi rule yang dipakai)
  - `signal_outcomes` (kode saham, tanggal sinyal, harga n+1/n+3/n+5 hari setelahnya, return %) — untuk tracking performa sinyal ke depannya
- Tambahkan validasi data: cek gap tanggal, candle yang hilang, atau nilai anomali (misal high < low) sebelum data dipakai untuk kalkulasi indikator.
- Desain data layer sebagai class/interface terpisah (misal `DataProvider`) sehingga fungsi fetch bisa diganti implementasinya nanti tanpa ubah kode di layer lain.

### 2. Indicator Layer
Hitung indikator teknikal dari data OHLCV pakai library seperti `pandas-ta` atau `ta`:
- RSI (14 hari default, tapi buat parameterizable)
- EMA cross (contoh: EMA9 vs EMA21, EMA20 vs EMA50)
- MACD + histogram
- Bollinger Bands (termasuk band width untuk deteksi squeeze)
- ATR (Average True Range) — untuk ukur volatilitas, dipakai juga nanti di risk management
- Volume relatif (volume hari ini dibanding rata-rata volume 20 hari)
- Support/resistance sederhana dari swing high-low N hari terakhir (misal 20/50 hari)
- (Opsional, kalau ada waktu) Deteksi pola candlestick dasar: bullish/bearish engulfing, hammer, doji

Setiap fungsi indikator harus independen dan testable (unit test dengan data dummy), supaya gampang divalidasi satu-satu.

### 3. Scoring Engine
- Kombinasikan semua indikator jadi satu skor akhir (0–100) pakai weighted scoring — bobot tiap indikator harus jadi parameter yang gampang diubah (config file atau dictionary), bukan hardcoded di logic.
- Output tiap skor harus menyertakan breakdown, contoh:
  ```json
  {
    "kode": "BBCA.JK",
    "tanggal": "2026-08-10",
    "skor_total": 78,
    "breakdown": [
      {"indikator": "RSI", "nilai": 32, "kontribusi": "keluar dari oversold", "skor": 20},
      {"indikator": "Volume", "nilai": "3.2x rata-rata", "kontribusi": "volume spike", "skor": 25},
      {"indikator": "EMA Cross", "nilai": "EMA9 > EMA21", "kontribusi": "golden cross", "skor": 20},
      {"indikator": "Resistance", "nilai": "breakout level 9200", "kontribusi": "breakout konfirmasi", "skor": 13}
    ]
  }
  ```
- Simpan hasil scoring harian ke tabel `signals` supaya bisa ditrack dan dibandingkan ke hasil aktual nanti.

### 4. Backtesting Engine
- Jalankan scoring rule ke data historis (minimal 2–3 tahun, banyak saham — jangan cuma satu saham/satu periode).
- Pakai **walk-forward testing**: bagi data jadi beberapa periode training/testing berurutan, bukan backtest sekali di satu rentang waktu penuh (supaya tidak overfit ke kondisi market tertentu).
- Metrics yang wajib dihitung:
  - Win rate (persentase sinyal yang menghasilkan return positif dalam N hari)
  - Average return per sinyal
  - Max drawdown
  - Distribusi return (jangan cuma rata-rata, karena bisa misleading kalau ada outlier)
- Buat laporan hasil backtest yang bisa diexport (CSV/HTML) untuk dianalisa manual.

### 5. Risk Management Layer
- Hitung suggested stop loss dan position size berdasarkan ATR (volatility-based sizing), bukan angka tetap.
- Tambahkan filter likuiditas: kalau volume rata-rata atau market cap saham di bawah threshold tertentu, beri warning "saham likuiditas rendah / rawan gorengan" di output sinyal.

### 6. Notification Layer (Telegram)
- Refactor bot Telegram supaya kirim skor + breakdown alasan, bukan cuma label bullish/bearish.
- Kalau memungkinkan, generate chart snapshot (pakai matplotlib/plotly, render ke image) yang nunjukin candlestick + indikator utama, dikirim bareng pesan teks.
- Format pesan idealnya:
  ```
  📊 BBCA.JK — Skor: 78/100
  
  RSI: 32 (keluar oversold)
  Volume: 3.2x rata-rata (spike)
  EMA9/EMA21: Golden cross
  Resistance: Breakout level 9200
  
  ⚠️ Ini bukan rekomendasi, evaluasi manual tetap wajib.
  ```

### 7. Performance Tracking / Journaling
- Setiap sinyal yang keluar dicatat ke `signal_outcomes`, lalu dicek otomatis performanya setelah 1, 3, 5 hari (bandingkan skor sinyal vs return aktual).
- Buat dashboard/report sederhana (bisa cukup script yang generate summary mingguan) buat lihat indikator mana yang paling berkontribusi ke sinyal yang akurat, supaya bisa terus dituning.

---

## Yang TIDAK termasuk di v1 pengembangan ini (biar scope tidak melebar)
- Deteksi ARA murni dari news/corporate action scraping (bisa jadi fase berikutnya).
- Eksekusi order otomatis ke broker.
- Real-time intraday signal (mulai dari daily timeframe dulu).
- Migrasi ke server/VPS (dilakukan setelah backtest & paper trading meyakinkan).

---

## Batasan yang Perlu Diketahui Developer/AI Koding
- `yfinance` untuk saham `.JK` punya keterbatasan: bisa delay, kadang data bolong untuk saham second/third liner, dan bukan API resmi (bisa berubah sewaktu-waktu). Untuk fase backtest & development ini oke, tapi desain `DataProvider` harus modular supaya gampang diganti nanti.
- Data intraday dari yfinance terbatas rentang waktunya (misal candle 1 menit cuma tersedia untuk ~7 hari terakhir) — jangan andalkan untuk backtest intraday jangka panjang.
- Semua rule/threshold (RSI oversold di bawah 30, misalnya) harus jadi parameter yang bisa diubah lewat config, bukan hardcoded, karena akan sering diubah selama proses backtest & tuning.

---

## Prioritas Implementasi (Urutan yang Disarankan)
1. Data layer + SQLite schema + validasi data
2. Indicator layer (mulai dari RSI, EMA cross, volume relatif dulu — paling simpel dan cepat divalidasi)
3. Scoring engine dasar (bobot manual dulu, belum perlu optimasi otomatis)
4. Backtesting engine — ini prioritas tinggi, jangan diskip atau ditunda
5. Risk management layer (ATR-based stop loss & position sizing)
6. Refactor notification layer ke Telegram
7. Performance tracking / journaling otomatis

## Catatan Tambahan
Sistem ini untuk keperluan screening & pengambilan keputusan trading, bukan untuk eksekusi otomatis. Prioritaskan transparansi (breakdown alasan skor) dan kemampuan diaudit dibanding kompleksitas fitur. Skor dan sinyal yang dihasilkan tetap harus dianggap sebagai alat bantu, bukan jaminan hasil.
