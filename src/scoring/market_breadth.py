"""
market_breadth.py — Market Breadth & Sector Context (Descriptive, Non-Predictive)

Fungsi Utama:
1. `get_market_breadth_context()` : Mengukur berapa banyak saham dari seluruh universe yang mengalami
                                   kondisi serupa hari ini (membedakan fenomena spesifik vs fenomena pasar-luas).
2. `get_sector_context()`         : Mengidentifikasi apakah saham lain di sektor yang sama juga aktif hari ini.

ATURAN KETAT:
- Murni informasi deskriptif kontekstual untuk membantu pemahaman manusia.
- TIDAK BOLEH digunakan untuk membuat filter otomatis, blacklist sektor, atau aturan scoring apapun.
"""

import requests
from typing import Dict, List, Any, Optional

# ─── Dynamic Sector Mapping ───────────────────────────────────────────────────
# Cache sector mapping untuk menghindari fetch berulang
_SECTOR_CACHE: Dict[str, str] = {}
_SECTOR_CACHE_LOADED = False


def _load_sector_mapping() -> Dict[str, str]:
    """
    Fetch sector mapping dari BEI API (idx.co.id).
    Fallback ke hardcoded minimal map jika API gagal.
    """
    global _SECTOR_CACHE, _SECTOR_CACHE_LOADED
    if _SECTOR_CACHE_LOADED:
        return _SECTOR_CACHE

    url = "https://www.idx.co.id/primary/ListedCompany/GetCompanyProfiles?draw=1&start=0&length=1500"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/',
    }

    try:
        sess = requests.Session()
        sess.get("https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/",
                 headers=headers, timeout=5)
        resp = sess.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            payload = resp.json()
            data_list = payload.get('data', [])
            mapping = {}
            for item in data_list:
                code = item.get('KodeEmiten', '').strip().upper()
                sector = item.get('SektorEmiten', '').strip()
                is_saham = item.get('EfekEmiten_Saham', True)
                if code and is_saham and len(code) == 4 and sector:
                    mapping[f"{code}.JK"] = sector
            if len(mapping) >= 100:
                print(f"[Market Breadth] ✅ Loaded {len(mapping)} sector mappings dari BEI.")
                _SECTOR_CACHE = mapping
                _SECTOR_CACHE_LOADED = True
                return _SECTOR_CACHE
    except Exception as err:
        print(f"[Market Breadth] Warning: Gagal fetch sector mapping dari BEI ({err}).")

    # Fallback: minimal hardcoded map untuk saham-saham utama
    _SECTOR_CACHE = _FALLBACK_SECTOR_MAP.copy()
    _SECTOR_CACHE_LOADED = True
    return _SECTOR_CACHE


# Fallback sector map (minimal, hanya saham blue-chip utama)
_FALLBACK_SECTOR_MAP = {
    'BBRI.JK': 'Perbankan & Keuangan',
    'BMRI.JK': 'Perbankan & Keuangan',
    'BBCA.JK': 'Perbankan & Keuangan',
    'BBNI.JK': 'Perbankan & Keuangan',
    'ANTM.JK': 'Komoditas & Tambang',
    'ADRO.JK': 'Komoditas & Tambang',
    'PTBA.JK': 'Komoditas & Tambang',
    'INCO.JK': 'Komoditas & Tambang',
    'PGAS.JK': 'Komoditas & Energi',
    'CTRA.JK': 'Properti & Infrastruktur',
    'DMAS.JK': 'Properti & Infrastruktur',
    'JSMR.JK': 'Properti & Infrastruktur',
    'PWON.JK': 'Properti & Infrastruktur',
    'UNVR.JK': 'Consumer Goods',
    'HMSP.JK': 'Consumer Goods',
    'GGRM.JK': 'Consumer Goods',
    'ICBP.JK': 'Consumer Goods',
    'INDF.JK': 'Consumer Goods',
    'ACES.JK': 'Ritel & Perdagangan',
    'RALS.JK': 'Ritel & Perdagangan',
    'ASII.JK': 'Otomotif & Konglomerasi',
    'TLKM.JK': 'Telekomunikasi',
    'EXCL.JK': 'Telekomunikasi',
    'ISAT.JK': 'Telekomunikasi',
    'AALI.JK': 'Perkebunan & Pertanian',
    'SGRO.JK': 'Perkebunan & Pertanian',
}


def get_sector_name(ticker: str) -> str:
    """Return nama sektor saham atau 'Lainnya'."""
    mapping = _load_sector_mapping()
    return mapping.get(ticker, 'Lainnya')


def get_market_breadth_context(
    condition_type: str,
    all_scanned_today: List[Dict[str, Any]],
    total_universe_count: int = None
) -> Optional[str]:
    """
    Hitung berapa banyak saham di universe yang mengalami kondisi yang sama hari ini.

    Args:
        total_universe_count: Jumlah total universe. Jika None, dihitung dari all_scanned_today.
    """
    matching_tickers = []
    for item in all_scanned_today:
        for tag in item.get('unusual_tags', []):
            if tag.get('tag_id') == condition_type:
                matching_tickers.append(item['ticker'])
                break

    count = len(matching_tickers)
    if count == 0:
        return None

    # Use actual universe count if not provided
    universe = total_universe_count if total_universe_count else max(len(all_scanned_today), 1)

    condition_names = {
        'VOLUME_SPIKE': 'volume spike',
        'RESISTANCE_BREAKOUT': 'breakout resistance',
        'SUPPORT_BREAKDOWN': 'breakdown support',
        'HIGH_RSI_RELATIVE': 'RSI tinggi',
        'LOW_RSI_RELATIVE': 'RSI rendah',
        'BB_SQUEEZE': 'BB squeeze',
        'MACD_BULLISH_CROSS': 'MACD bullish cross',
        'MACD_BEARISH_CROSS': 'MACD bearish cross',
        'EMA50_FAR_ABOVE': 'harga jauh di atas EMA50',
        'EMA50_FAR_BELOW': 'harga jauh di bawah EMA50',
        'ATR_EXPANSION': 'volatilitas meningkat (ATR expansion)',
        'GAP_UP': 'gap up',
        'GAP_DOWN': 'gap down',
    }
    c_name = condition_names.get(condition_type, 'kondisi di luar kebiasaan')

    if count >= 6:
        return f"Konteks pasar: {count} dari {universe} saham di universe mengalami {c_name} hari ini (pola pasar luas)"
    else:
        return f"Konteks pasar: {count} dari {universe} saham di universe mengalami {c_name} hari ini (cenderung spesifik ke saham ini)"


def get_sector_context(
    target_ticker: str,
    all_scanned_today: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Tampilkan apakah saham lain di sektor yang sama juga menunjukkan aktivitas di luar kebiasaan hari ini.
    """
    target_sector = get_sector_name(target_ticker)
    if target_sector == 'Lainnya':
        return None

    peers_active = []
    for item in all_scanned_today:
        t = item['ticker']
        if t != target_ticker and get_sector_name(t) == target_sector:
            peers_active.append(t.replace('.JK', ''))

    if not peers_active:
        return None

    peer_str = ", ".join(peers_active[:3])  # Limit display to top 3 peers
    return f"Konteks sektor: {len(peers_active)} saham lain di {target_sector} ({peer_str}) juga aktif hari ini"
