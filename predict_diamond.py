import pandas as pd

# 시뮬레이션된 다이아몬드 가격 예측 함수
# 실제로는 학습된 모델(예: LinearRegression 모델)을 불러와서 사용합니다.
def predict_price_with_model(carat, cut, color, clarity):
    """
    다이아몬드 특성을 기반으로 가격을 예측하는 함수입니다.
    이 함수는 학습된 모델을 시뮬레이션합니다.
    """
    # 범주형 데이터를 숫자형으로 변환하는 매핑 (실제 모델 학습 시 사용한 방식)
    cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    color_mapping = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
    clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

    # 입력값을 기반으로 가중치를 계산 (간단한 시뮬레이션)
    cut_score = cut_mapping.get(cut, 0)
    color_score = color_mapping.get(color, 0)
    clarity_score = clarity_mapping.get(clarity, 0)

    # 실제 학습된 모델은 이러한 특성들의 조합에 따른 복잡한 공식을 사용합니다.
    # 여기서는 캐럿이 가장 큰 영향을 준다는 점을 반영해 시뮬레이션합니다.
    predicted_price = 2000 * carat**2 + (cut_score * 50) + (color_score * 100) + (clarity_score * 150)
    
    return predicted_price

# 사용자로부터 다이아몬드 특성 입력받기
try:
    print("다이아몬드 가격 예측을 시작합니다.")
    print("----------------------------")
    
    carat_input = float(input("캐럿(Carat) 값을 입력하세요 (예: 0.5): "))
    
    cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut_input = input(f"컷(Cut) 등급을 입력하세요 {cut_options}: ")
    if cut_input not in cut_options:
        raise ValueError(f"유효하지 않은 컷 등급입니다. {cut_options} 중 하나를 선택하세요.")

    color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    color_input = input(f"컬러(Color) 등급을 입력하세요 {color_options}: ")
    if color_input not in color_options:
        raise ValueError(f"유효하지 않은 컬러 등급입니다. {color_options} 중 하나를 선택하세요.")

    clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity_input = input(f"투명도(Clarity) 등급을 입력하세요 {clarity_options}: ")
    if clarity_input not in clarity_options:
        raise ValueError(f"유효하지 않은 투명도 등급입니다. {clarity_options} 중 하나를 선택하세요.")

    # 예측 함수 실행
    predicted_price = predict_price_with_model(
        carat=carat_input,
        cut=cut_input,
        color=color_input,
        clarity=clarity_input
    )

    print("----------------------------")
    print(f"예상 다이아몬드 가격은 약 ${predicted_price:,.2f} 입니다.")

except ValueError as e:
    print(f"\n오류가 발생했습니다: {e}")
    print("숫자를 정확히 입력하거나 제시된 등급 중 하나를 선택해주세요.")
