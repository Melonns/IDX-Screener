"""
news_intelligence_manager.py — Unified News & Sector Intelligence Manager (Phase 5)

Menggabungkan 3 Pilar Intelijen Berita & Sektor menjadi 1 Engine Terpadu:
1. Pengumuman Keterbukaan Informasi Resmi IDX (idx.co.id GetNewsSearch)
2. Berita Media Keuangan Real-Time (CNBC Indonesia & Kontan RSS Feeds)
3. Profil Sektor & Sector Trend Filtering (Automated Blacklisting untuk Sektor Tertekan)
"""

import sys, os, time, re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

_HERE = Path(__file__).parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from data.database import DatabaseManager
import config as app_config

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
    'X-Requested-With': 'XMLHttpRequest',
}

# Mapping Sektor Resmi & Status Headwind
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
    'ACES.JK': 'Ritel & Perdagangan',
    'RALS.JK': 'Ritel & Perdagangan',
    'ASII.JK': 'Otomotif & Konglomerasi',
    'TLKM.JK': 'Telekomunikasi',
    'EXCL.JK': 'Telekomunikasi',
    'ISAT.JK': 'Telekomunikasi',
}

# Sektor yang mengalami tekanan struktural (Blacklisted by default unless strong catalyst)
BLOCKED_SECTORS = {'Consumer Goods', 'Komoditas & Energi'}


class NewsIntelligenceManager:
    """
    Manager Terpadu untuk Berita, Aksi Korporasi, & Intelijen Sektor.
    """
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.tickers = set([t.replace('.JK', '') for t in db.get_tickers()])

    def get_stock_sector_status(self, ticker: str) -> Dict[str, Any]:
        """
        Check if stock belongs to a favorable sector or blocked sector.
        """
        sector = SECTOR_MAP.get(ticker, 'Lainnya')
        is_blocked = sector in BLOCKED_SECTORS
        return {
            'ticker': ticker,
            'sector': sector,
            'is_blocked': is_blocked,
            'status': 'HEADWIND_BLOCKED' if is_blocked else 'SECTOR_FAVORABLE'
        }

    def sync_all_news_sources(self) -> Dict[str, int]:
        """
        Sync both IDX Official Announcements AND Financial Media RSS Feeds.
        """
        idx_count = self._sync_idx_announcements()
        rss_count = self._sync_rss_news()
        return {'idx_announcements': idx_count, 'rss_media': rss_count}

    def _sync_idx_announcements(self, max_pages: int = 20) -> int:
        session = requests.Session()
        session.headers.update(HEADERS)
        try: session.get('https://www.idx.co.id/', timeout=5)
        except Exception: pass

        rows = []
        for page in range(1, max_pages + 1):
            url = f"https://www.idx.co.id/primary/NewsAnnouncement/GetNewsSearch?pageNumber={page}&pageSize=50"
            try:
                res = session.get(url, timeout=10)
                if res.status_code != 200: continue
                items = res.json().get('Items', [])
                if not items: break

                for item in items:
                    ann_id  = str(item.get('ItemId', item.get('Id', '')))
                    pub_d   = item.get('PublishedDate', '')[:10]
                    title   = item.get('Title', '')
                    tags    = item.get('Tags', '')
                    summary = item.get('Summary', '')
                    ticker  = self._extract_ticker(title, tags)

                    rows.append((ann_id, pub_d, ticker, title, tags, summary))
                time.sleep(0.2)
            except Exception:
                continue
        return self.db.save_idx_announcements(rows)

    def _sync_rss_news(self) -> int:
        rss_feeds = [
            ("CNBC Indonesia Market", "https://www.cnbcindonesia.com/market/rss"),
            ("Kontan Investasi", "https://investasi.kontan.co.id/rss"),
        ]
        rows = []
        for name, url in rss_feeds:
            try:
                res = requests.get(url, headers=HEADERS, timeout=10)
                if res.status_code == 200:
                    root = ET.fromstring(res.content)
                    channel = root.find('channel')
                    items = channel.findall('item') if channel is not None else []

                    for item in items:
                        title   = item.findtext('title', '')
                        pub_d   = item.findtext('pubDate', '')[:16]
                        summary = item.findtext('description', '')
                        ticker  = self._extract_ticker(title, summary)
                        ann_id  = f"RSS_{hash(title)}"

                        rows.append((ann_id, pub_d, ticker, title, name, summary))
            except Exception:
                continue
        return self.db.save_idx_announcements(rows)

    def _extract_ticker(self, title: str, text: str) -> str:
        content = f"{title} {text}"
        found = re.findall(r'\b[A-Z]{4}\b', content)
        for code in found:
            if code in self.tickers:
                return f"{code}.JK"
        return None


if __name__ == '__main__':
    db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    mgr = NewsIntelligenceManager(db)
    counts = mgr.sync_all_news_sources()
    print(f"Berhasil mengabungkan data berita & pengumuman: {counts}")
