
# pipeline_e2e_mini.py
# 풀 파이프라인 미니 (수집→검증→전처리→학습/평가→조건부 배포→알림→메트릭 저장)
# ▶ CSV/DB/API 중 하나를 골라 수집하고, 검증→전처리→학습/평가→조건부 배포→슬랙 알림까지
import os, json
from pathlib import Path
import pandas as pd
from joblib import dump, load
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import requests

# ───────────── 설정/경로 ─────────────
load_dotenv()  # .env 읽기
SOURCE_TYPE = os.getenv("SOURCE_TYPE", "csv")  # csv|db|api

PROD_DIR = Path("production"); PROD_DIR.mkdir(exist_ok=True)
PROD_MODEL = PROD_DIR / "model.joblib"
PROD_METRICS = PROD_DIR / "metrics.json"
RUNS_DIR = Path("runs"); RUNS_DIR.mkdir(exist_ok=True)

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

def notify(msg: str):
    print(msg)
    if SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=8)
        except Exception as e:
            print("[Slack 실패]", e)

# ───────────── 1) 수집(ingest) ─────────────
def ingest_csv() -> pd.DataFrame:
    path = os.getenv("CSV_PATH")
    assert path, "CSV_PATH가 .env에 필요합니다."
    return pd.read_csv(path)

def ingest_db() -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    db_url = os.getenv("DB_URL"); sql = os.getenv("DB_SQL")
    assert db_url and sql, "DB_URL, DB_SQL이 .env에 필요합니다."
    eng = create_engine(db_url)
    with eng.begin() as conn:
        return pd.read_sql(text(sql), conn)

def ingest_api() -> pd.DataFrame:
    url = os.getenv("API_URL"); timeout = int(os.getenv("API_TIMEOUT", "15"))
    assert url, "API_URL이 .env에 필요합니다."
    r = requests.get(url, timeout=timeout); r.raise_for_status()
    data = r.json()
    # 상황에 맞게 파싱:
    # - 이미 레코드 리스트면 바로 DataFrame
    # - 딕셔너리면 내부 키 선택 필요
    return pd.DataFrame(data)

def ingest() -> pd.DataFrame:
    if SOURCE_TYPE == "csv": return ingest_csv()
    if SOURCE_TYPE == "db":  return ingest_db()
    if SOURCE_TYPE == "api": return ingest_api()
    raise ValueError(f"지원하지 않는 SOURCE_TYPE: {SOURCE_TYPE}")

# ───────────── 2) 검증(validate) ─────────────
def validate(df: pd.DataFrame, required_cols: list[str]):
    assert len(df) >= 50, f"행 수 이상: {len(df)}"
    for c in required_cols:
        assert c in df.columns, f"필수 컬럼 누락: {c}"
    # (선택) 컬럼별 결측/범위 체크도 여기에 추가 가능
    # 예: assert df["age"].between(0, 120).all(), "age 범위 오류"

# ───────────── 3) 전처리(preprocess) + 모델 ─────────────
def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])
    return Pipeline([("prep", preprocess),
                     ("clf",  LogisticRegression(max_iter=1000))])

# ───────────── 메트릭 저장/로드 ─────────────
def save_metrics(path: Path, metrics: dict):
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

def load_metrics(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

# ───────────── 메인 플로우 ─────────────
def main():
    # 1) 수집
    df = ingest()

    # ⚠️ 여기만 도메인별로 맞추면 끝! (예: 음악/영화/이커머스)
    # 예시 스키마: age(수치), country(범주), gender(범주), avg_session(수치), label(이진)
    required = ["age","country","gender","avg_session","label"]
    validate(df, required_cols=required)

    num_cols = ["age","avg_session"]
    cat_cols = ["country","gender"]
    label_col = "label"

    X = df[num_cols + cat_cols]
    y = df[label_col]

    pipe = build_pipeline(num_cols, cat_cols)

    # 4) 학습 & 평가
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "accuracy": float(accuracy_score(yte, pred)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte))
    }
    save_metrics(RUNS_DIR / "last_metrics.json", metrics)

    # 5) 조건부 배포 (기존보다 좋아야 교체)
    old = load_metrics(PROD_METRICS)
    old_auc = old["roc_auc"] if old else -1.0

    if metrics["roc_auc"] > old_auc:
        dump(pipe, PROD_MODEL)
        save_metrics(PROD_METRICS, metrics)
        notify(f"✅ 새 모델 배포 (AUC {old_auc:.3f} → {metrics['roc_auc']:.3f})")
    else:
        notify(f"ℹ️ 기존 유지 (AUC {old_auc:.3f} vs {metrics['roc_auc']:.3f})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(f"❌ 파이프라인 실패: {e}")
        raise
