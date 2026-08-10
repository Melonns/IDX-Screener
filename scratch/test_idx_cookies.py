"""
test_idx_cookies.py — Pengujian Akses IDX API via Session & Alternatif yfinance
"""
import requests
import json
import yfinance as yf

print("="*75)
print("  TEST 1: Requests Session ke idx.co.id")
print("="*75)

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
})

try:
    # 1. Main page to get cookies
    r1 = session.get('https://www.idx.co.id/', timeout=10)
    print(f"  Main page status: {r1.status_code}, Cookies: {dict(session.cookies)}")
    
    # 2. News API call with session
    session.headers.update({
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.idx.co.id/id/berita/pengumuman/',
        'X-Requested-With': 'XMLHttpRequest',
    })
    
    url_news = "https://www.idx.co.id/primary/NewsAnnouncement/GetNewsSearch?pageNumber=1&pageSize=5"
    r2 = session.get(url_news, timeout=10)
    print(f"  News API status: {r2.status_code}")
    if r2.status_code == 200:
        data = r2.json()
        print(f"  ItemCount: {data.get('ItemCount', 'N/A')}")
        if data.get('Items'):
            print(f"  Sample Title: {data['Items'][0].get('Title')}")

    # 3. Corporate Action API with session
    url_ca = "https://www.idx.co.id/primary/ListedCompany/GetCorporateAction?indexFrom=1&pageSize=5"
    r3 = session.get(url_ca, timeout=10)
    print(f"  Corporate Action API status: {r3.status_code}")

except Exception as e:
    print(f"  Session test error: {e}")


print("\n" + "="*75)
print("  TEST 2: Alternatif yfinance (Dividen & Split Historis)")
print("="*75)

for ticker_code in ["BBCA.JK", "TLKM.JK", "ASII.JK"]:
    try:
        t = yf.Ticker(ticker_code)
        divs = t.dividends
        splits = t.splits
        print(f"  {ticker_code:<10} | Dividen record count: {len(divs)} | Split record count: {len(splits)}")
        if not divs.empty:
            last_div = divs.iloc[-1]
            last_div_date = divs.index[-1].strftime('%Y-%m-%d')
            print(f"               Dividen Terakhir: {last_div_date} (Rp {last_div:.2f}/saham)")
    except Exception as e:
        print(f"  {ticker_code:<10} | Error: {e}")
