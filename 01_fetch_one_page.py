# 01_fetch_one_page.py
import requests, pandas as pd
from bs4 import BeautifulSoup

URL = "https://quotes.toscrape.com/"
HDRS = {"User-Agent": "Mozilla/5.0 (practice-crawl; +https://example.com)"}

r = requests.get(URL, headers=HDRS, timeout=10)
r.raise_for_status()
soup = BeautifulSoup(r.text, "lxml")

rows = []
for q in soup.select("div.quote"):
    text = q.select_one("span.text").get_text(strip=True)
    author = q.select_one("small.author").get_text(strip=True)
    tags = [t.get_text(strip=True) for t in q.select("div.tags a.tag")]
    rows.append({"text": text, "author": author, "tags": "|".join(tags)})

df = pd.DataFrame(rows)
df.to_csv("quotes_page1.csv", index=False)
print("✅ saved quotes_page1.csv, rows:", len(df))
print(df.head(3))
