# 02_fetch_all_pages.py
import time, requests, pandas as pd
from urllib.parse import urljoin
from bs4 import BeautifulSoup

BASE = "https://quotes.toscrape.com/"
HDRS = {"User-Agent": "Mozilla/5.0 (practice-crawl; +https://example.com)"}

def crawl_all(max_pages=999):
    url = BASE
    all_rows = []
    seen = 0
    for _ in range(max_pages):
        r = requests.get(url, headers=HDRS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        rows = []
        for q in soup.select("div.quote"):
            text = q.select_one("span.text").get_text(strip=True)
            author = q.select_one("small.author").get_text(strip=True)
            tags = [t.get_text(strip=True) for t in q.select("div.tags a.tag")]
            rows.append({"text": text, "author": author, "tags": "|".join(tags)})
        all_rows.extend(rows)
        seen += len(rows)
        print(f"page collected, total rows={seen}")

        nxt = soup.select_one("li.next a")
        if not nxt:
            break
        url = urljoin(url, nxt["href"])
        time.sleep(0.5)  # 예의상 살짝 대기

    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    df = crawl_all()
    df.to_csv("quotes_all.csv", index=False)
    print("✅ saved quotes_all.csv, rows:", len(df))
    print(df.sample(5))
