"""
idx_news_collector.py — Scraper & Collector Keterbukaan Informasi Resmi IDX (Phase 5)

Menarik pengumuman resmi dari API GetNewsSearch idx.co.id,
mendeteksi kode saham dari Judul/Tags, dan menyimpannya ke tabel idx_announcements.
"""
import sys, os, time, re
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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
    'Referer': 'https://www.idx.co.id/id/berita/pengumuman/',
    'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
    'X-Requested-With': 'XMLHttpRequest',
}


class IDXNewsCollector:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.tickers = set([t.replace('.JK', '') for t in db.get_tickers()])

    def collect_news(self, max_pages: int = 30, page_size: int = 50) -> int:
        print(f"Collecting IDX News & Announcements ({max_pages} pages x {page_size} items)...")
        session = requests.Session()
        session.headers.update(HEADERS)
        
        # Warmup session cookie
        try:
            session.get('https://www.idx.co.id/', timeout=5)
        except Exception:
            pass

        rows_to_save = []

        for page in tqdm(range(1, max_pages + 1), desc="Fetching News Pages"):
            url = f"https://www.idx.co.id/primary/NewsAnnouncement/GetNewsSearch?pageNumber={page}&pageSize={page_size}"
            try:
                res = session.get(url, timeout=10)
                if res.status_code != 200:
                    print(f"Page {page} returned status {res.status_code}")
                    continue

                data = res.json()
                items = data.get('Items', [])
                if not items:
                    break

                for item in items:
                    ann_id = str(item.get('ItemId', item.get('Id', '')))
                    pub_date = item.get('PublishedDate', '')
                    date_str = pub_date[:10] if pub_date else ''
                    title = item.get('Title', '')
                    tags = item.get('Tags', '')
                    summary = item.get('Summary', '')

                    # Extract ticker code using regex or tag match
                    ticker = self._extract_ticker(title, tags)

                    rows_to_save.append((ann_id, date_str, ticker, title, tags, summary))

                time.sleep(0.3)  # Gentle rate limit

            except Exception as exc:
                print(f"Error fetching page {page}: {exc}")

        saved_count = self.db.save_idx_announcements(rows_to_save)
        print(f"Berhasil menyimpan {saved_count} pengumuman resmi IDX ke database.")
        return saved_count

    def _extract_ticker(self, title: str, tags: str) -> str:
        """Deteksi 4-letter ticker code dari judul atau tags."""
        text = f"{title} {tags}"
        # Match pattern [BBCA] or BBCA
        found = re.findall(r'\b[A-Z]{4}\b', text)
        for code in found:
            if code in self.tickers:
                return f"{code}.JK"
        return None


if __name__ == '__main__':
    db_path = os.path.join(app_config.DATA_DIR, 'idx_screener.db')
    db = DatabaseManager(db_path)
    collector = IDXNewsCollector(db)
    collector.collect_news(max_pages=30, page_size=50)
