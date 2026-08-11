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

from typing import Dict, List, Any, Optional

# Mapping Sektor Informasi (Fakta Metadata)
SECTOR_MAP = {
    'BBRI.JK': 'Perbankan & Keuangan',
    'BMRI.JK': 'Perbankan & Keuangan',
    'BBCA.JK': 'Perbankan & Keuangan',
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
    return SECTOR_MAP.get(ticker, 'Lainnya')


def get_market_breadth_context(
    condition_type: str,
    all_scanned_today: List[Dict[str, Any]],
    total_universe_count: int = 45
) -> Optional[str]:
    """
    Hitung berapa banyak saham di universe yang mengalami kondisi yang sama hari ini.

    Returns:
        str: Kalimat deskriptif konteks pasar atau None
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

    condition_names = {
        'VOLUME_SPIKE': 'volume spike',
        'RESISTANCE_BREAKOUT': 'breakout resistance',
        'HIGH_RSI_RELATIVE': 'RSI tinggi',
        'LOW_RSI_RELATIVE': 'RSI rendah',
        'BB_SQUEEZE': 'BB squeeze'
    }
    c_name = condition_names.get(condition_type, 'kondisi di luar kebiasaan')

    if count >= 6:
        return f"Konteks pasar: {count} dari {total_universe_count} saham di universe mengalami {c_name} hari ini (pola pasar luas)"
    else:
        return f"Konteks pasar: {count} dari {total_universe_count} saham di universe mengalami {c_name} hari ini (cenderung spesifik ke saham ini)"


def get_sector_context(
    target_ticker: str,
    all_scanned_today: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Tampilkan apakah saham lain di sektor yang sama juga menunjukkan aktivitas di luar kebiasaan hari ini.

    Returns:
        str: Kalimat deskriptif sektor atau None
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

    peer_str = ", ".join(peers_active[:3]) # Limit display to top 3 peers
    return f"Konteks sektor: {len(peers_active)} saham lain di {target_sector} ({peer_str}) juga aktif hari ini"
