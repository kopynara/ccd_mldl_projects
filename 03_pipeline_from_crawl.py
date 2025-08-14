# 03_pipeline_from_crawl.py
# ▶ 크롤링→검증→전처리→학습/평가→(조건부)배포→슬랙 알림
import os, json, time, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import dump, load

import random, time, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session():
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=1.0,  # 1s, 2s, 4s 지수 백오프
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    s.headers.update({"User-Agent": "mldl-practice-crawl/1.0 (contact: you@example.com)"})
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

S = make_session()


# ───── 설정/경로 ─────
load_dotenv()
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
HDRS = {"User-Agent": "Mozilla/5.0 (practice-crawl; +https://example.com)"}

PROD_DIR = Path("production"); PROD_DIR.mkdir(exist_ok=True)
PROD_MODEL = PROD_DIR / "model.joblib"
PROD_METRICS = PROD_DIR / "metrics.json"

def notify(msg: str):
    print(msg)
    if SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=8)
        except Exception as e:
            print("[Slack 실패]", e)

# ───── 1) 수집(크롤링) ─────
def crawl_all(max_pages=999):
    BASE = "https://quotes.toscrape.com/"
    url = BASE
    all_rows = []
    for _ in range(max_pages):
        r = requests.get(url, headers=HDRS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for q in soup.select("div.quote"):
            text = q.select_one("span.text").get_text(strip=True)
            author = q.select_one("small.author").get_text(strip=True)
            tags = [t.get_text(strip=True) for t in q.select("div.tags a.tag")]
            all_rows.append({"text": text, "author": author, "tags": "|".join(tags)})
        nxt = soup.select_one("li.next a")
        if not nxt:
            break
        url = urljoin(url, nxt["href"])
        time.sleep(2)
    return pd.DataFrame(all_rows)

# ───── 2) 검증 ─────
def validate(df: pd.DataFrame):
    assert len(df) >= 50, f"행수 이상: {len(df)}"
    for c in ["text","author","tags"]:
        assert c in df.columns, f"필수 컬럼 누락: {c}"
    # 추가 검증 아이디어: 중복 제거, 빈 텍스트 체크 등

# ───── 3) 전처리 + 모델 ─────
def build_pipeline(num_cols, cat_cols):
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("prep", preprocess),
                     ("clf", LogisticRegression(max_iter=1000))])

def main():
    # 1) 수집
    raw = crawl_all()
    raw.to_csv("quotes_latest.csv", index=False)  # 캐시/디버깅용

    # 파생 특성(간단): 글 길이, 태그 개수
    df = raw.copy()
    df["text_len"] = df["text"].str.len()   
    df["tag_count"] = (
    df["tags"].fillna("")
      .str.split("|", regex=False)     # '|'를 그냥 문자로 분리
      .apply(lambda lst: sum(1 for t in lst if t))
)    
    # (데모용) 라벨: 긴 글이면 1, 아니면 0
    df["label"] = (df["text_len"] > 100).astype(int)

    # 2) 검증
    validate(df)

    # 3) 전처리 + 4) 학습/평가
    num_cols = ["text_len","tag_count"]
    cat_cols = ["author"]
    X = df[num_cols + cat_cols]
    y = df["label"]

    pipe = build_pipeline(num_cols, cat_cols)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc":  float(roc_auc_score(yte, proba)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte))
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/last_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 5) (조건부)배포
    if PROD_METRICS.exists():
        old_auc = json.loads(PROD_METRICS.read_text(encoding="utf-8"))["roc_auc"]
    else:
        old_auc = -1.0

    if metrics["roc_auc"] > old_auc:
        dump(pipe, PROD_MODEL)
        PROD_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        notify(f"✅ 새 모델 배포 (AUC {old_auc:.3f} → {metrics['roc_auc']:.3f})")
    else:
        notify(f"ℹ️ 기존 유지 (AUC {old_auc:.3f} vs {metrics['roc_auc']:.3f})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(f"❌ 파이프라인 실패: {e}")
        raise
