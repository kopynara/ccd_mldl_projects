# BOOT-10_모델_저장_및_로드_실습.py

# 1. 필요한 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import joblib # 모델을 저장하고 로드하기 위한 라이브러리

# 경고 메시지 무시
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """
    타이타닉 데이터를 학습하고 최적의 CatBoost 모델을 파일로 저장하는 함수.
    """
    print("### 학습된 모델 저장 실습 시작 ###")

    # 2. 데이터셋 로드
    df_train = pd.read_csv('../../titanic_project/titanic_train.csv')

    # 3. 데이터 전처리
    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
    df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
    df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

    # 특성과 레이블 분리
    X = df_train.drop('Survived', axis=1)
    y = df_train['Survived']

    # 학습 및 테스트 데이터 분리 (모델 검증을 위해)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 데이터 전처리 파이프라인 구축 (공통)
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

    # 5. 최적 모델 정의 및 파이프라인 생성
    # 이전 튜닝 단계에서 찾은 최적의 CatBoost 파라미터를 적용
    best_model = CatBoostClassifier(iterations=350,
                                    learning_rate=0.1,
                                    depth=6,
                                    silent=True,
                                    random_state=42,
                                    verbose=0)
    best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', best_model)])

    print("\n--- CatBoost 모델 학습 시작 ---")
    best_pipeline.fit(X_train, y_train)
    print("--- 모델 학습 완료 ---")

    # 6. 학습된 모델을 파일로 저장
    model_filename = 'best_titanic_model.pkl'
    joblib.dump(best_pipeline, model_filename)
    print(f"\n✅ 모델이 '{model_filename}' 파일로 성공적으로 저장되었습니다.")

    return X_test, y_test # 예측 검증을 위해 테스트 데이터 반환

def load_and_predict(model_filename, X_test):
    """
    저장된 모델 파일을 로드하고 예측을 수행하는 함수.
    """
    print("\n\n--- 저장된 모델 파일 로드 및 예측 실습 ---")

    # 7. 저장된 모델 파일을 로드
    loaded_pipeline = joblib.load(model_filename)
    print(f"✅ 모델이 '{model_filename}' 파일에서 성공적으로 로드되었습니다.")

    # 8. 로드된 모델로 예측 수행
    y_pred_loaded = loaded_pipeline.predict(X_test)
    print(f"\n로드된 모델을 사용하여 테스트 데이터의 첫 5개 예측값: {y_pred_loaded[:5]}")

    return y_pred_loaded

if __name__ == '__main__':
    # 학습 및 저장
    X_test_data, y_test_data = train_and_save_model()
    
    # 로드 및 예측
    model_file = 'best_titanic_model.pkl'
    y_pred_results = load_and_predict(model_file, X_test_data)
    
    print(f"실제 정답(y_test)의 첫 5개 값: {y_test_data.values[:5]}")
