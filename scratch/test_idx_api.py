"""
test_idx_api.py — Investigasi & Pengujian Endpoint API Resmi IDX (idx.co.id)

Tujuan: Menguji berbagai endpoint resmi IDX dengan custom User-Agent & Header
untuk menemukan endpoint yang aktif (Status 200 OK) untuk Data Corporate Action & Keterbukaan Informasi.
"""
import requests
import json

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.idx.co.id/',
    'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
}

endpoints_to_test = [
    # 1. News Announcement / Keterbukaan Informasi (User mengonfirmasi ini AKTIF!)
    ("News Announcement Search", "https://www.idx.co.id/primary/NewsAnnouncement/GetNewsSearch?pageNumber=1&pageSize=5"),
    
    # 2. Corporate Action Calendar (Coba parameter tambahan / path variasi)
    ("Corporate Action (Primary)", "https://www.idx.co.id/primary/ListedCompany/GetCorporateAction?indexFrom=1&pageSize=10"),
    ("Corporate Action (API V1)", "https://www.idx.co.id/idx.my.id/api/v1/CorporateAction/GetCorporateAction?indexFrom=1&pageSize=10"),
    
    # 3. Dividend Specific Calendar
    ("Dividen Calendar", "https://www.idx.co.id/primary/ListedCompany/GetDividendHistory?indexFrom=1&pageSize=10"),
    
    # 4. KSEI / Shareholder composition
    ("Company Profile / Financials", "https://www.idx.co.id/primary/ListedCompany/GetCompanyProfiles?indexFrom=1&pageSize=5"),
]

print("="*75)
print("  PENGUJIEN ENDPOINT API RESMI BURSA EFEK INDONESIA (IDX)")
print("="*75)

for name, url in endpoints_to_test:
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        status = res.status_code
        if status == 200:
            data = res.json()
            item_count = len(data) if isinstance(data, list) else data.get('ItemCount', data.get('total', 'N/A'))
            print(f"  [✅ 200 OK] {name:<35} | Items: {item_count}")
            # Show sample keys/title if available
            if isinstance(data, dict) and 'Items' in data and len(data['Items']) > 0:
                first = data['Items'][0]
                sample_str = first.get('Title', first.get('title', str(first)[:80]))
                print(f"       Sample: {sample_str[:70]}")
        else:
            print(f"  [❌ {status}] {name:<35} | Error/Unavailable")
    except Exception as e:
        print(f"  [⚠️ ERR] {name:<35} | Exception: {str(e)[:50]}")
