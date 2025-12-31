from sklearn import preprocessing # 데이터 전처리를 위한 모듈 (인코딩, 스케일링 등)
from sklearn.preprocessing import LabelEncoder # 문자를 숫자로 변환(레이블 인코딩)하기 위한 클래스
from sklearn.model_selection import train_test_split # 학습 데이터와 테스트 데이터를 분리하는 함수
from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 분류 모델
from sklearn.ensemble import RandomForestClassifier # 앙상블 학습 기법 중 하나인 랜덤 포레스트 모델
from sklearn.linear_model import LogisticRegression # 분류를 위한 선형 모델인 로지스틱 회귀 모델
from sklearn.metrics import accuracy_score # 모델의 예측 정확도를 계산하는 평가 함수
from sklearn.model_selection import KFold # 교차 검증을 위한 K-Fold 객체
from sklearn.model_selection import cross_val_score # 교차 검증 점수를 간편하게 계산하는 함수
from sklearn.model_selection import GridSearchCV # 하이퍼 파라미터 튜닝과 교차 검증을 동시에 수행하는 객체
import numpy as np # 수치 계산을 위한 핵심 라이브러리
import pandas as pd # 표 형태의 데이터 처리를 위한 라이브러리
import os # 파일 경로 조작 등 운영체제 기능 사용을 위한 모듈

# 경고 메시지 무시 설정
# 모델 학습 시 버전 차이 등으로 인해 발생하는 불필요한 경고를 숨겨줍니다.
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. 데이터 로드 과정
    # 현재 실행 중인 파일의 디렉토리 경로를 가져옵니다.
    path = os.path.dirname(__file__)
    # 경로와 파일명을 결합하여 CSV 파일의 전체 경로를 생성합니다.
    load_file = os.path.join(path, 'titanic_train.csv')
    
    # pandas를 이용하여 CSV 데이터를 DataFrame으로 로드합니다.
    titanic_df = pd.read_csv(load_file)
    print("### 데이터 로드 완료 ###")
    # 로드된 데이터의 상위 3개 행을 출력하여 구조를 확인합니다.
    print(titanic_df.head(3))
    
    # 2. 전처리 함수 정의
    
    # 결측치(NaN)를 처리하는 함수
    def fillna(df):
        # Age(나이)의 결측치는 전체 평균값으로 채웁니다. (inplace=True로 원본 수정)
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        # Cabin(선실 번호)의 결측치는 'N'으로 채웁니다.
        df['Cabin'].fillna('N', inplace=True)
        # Embarked(정박 항구)의 결측치는 'N'으로 채웁니다.
        df['Embarked'].fillna('N', inplace=True)
        # Fare(요금)의 결측치는 0으로 채웁니다.
        df['Fare'].fillna(0, inplace=True)
        return df

    # 모델 학습에 불필요한 피처를 제거하는 함수
    def drop_features(df):
        # PassengerId(승객ID), Name(이름), Ticket(티켓번호)는 생존 예측과 관련성이 낮아 제거합니다.
        df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
        return df

    # 문자열 카테고리 피처를 숫자형으로 변환(레이블 인코딩)하는 함수
    def format_features(df):
        # Cabin(선실)의 경우 등급을 나타내는 첫 글자가 중요하므로 앞문자만 추출합니다.
        df['Cabin'] = df['Cabin'].str[:1]
        
        # 인코딩할 대상 피처 리스트
        features = ['Cabin', 'Sex', 'Embarked']
        for feature in features:
            le = LabelEncoder() # LabelEncoder 객체 생성
            le = le.fit(df[feature]) # 해당 피처의 데이터로 인코딩 규칙 학습
            df[feature] = le.transform(df[feature]) # 규칙에 따라 데이터를 숫자로 변환
        return df

    # 위의 전처리 단계들을 순차적으로 실행하는 전체 전처리 함수
    def transform_features(df):
        df = fillna(df)          # 1. 결측치 처리
        df = drop_features(df)   # 2. 불필요 피처 삭제
        df = format_features(df) # 3. 포맷팅 및 인코딩
        return df

    # 3. 데이터 전처리 및 분리
    # 레이블(Target, 정답) 데이터 분리: 'Survived' 컬럼
    y_titanic_df = titanic_df['Survived']
    # 피처(Input, 학습) 데이터 분리: 'Survived'를 제외한 나머지 컬럼
    X_titanic_df = titanic_df.drop('Survived', axis=1)
    
    # 정의된 전처리 함수를 수행하여 X 데이터를 변환합니다.
    X_titanic_df = transform_features(X_titanic_df)

    # 학습 데이터와 테스트 데이터를 분리합니다. (테스트 비율 20%, 난수 시드 11)
    # y_titanic_df를 포함해야 학습 시 정답 데이터도 함께 나뉩니다.
    X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)
    print("\n### 데이터 전처리 및 분리 완료 ###")

    # 4. 모델 및 파라미터 정의
    
    # 각기 다른 알고리즘을 사용하는 분류 모델 객체 생성
    dt_clf = DecisionTreeClassifier(random_state=11) # 의사결정 트리 (random_state로 결과 고정)
    rf_clf = RandomForestClassifier(random_state=11) # 랜덤 포레스트
    lr_clf = LogisticRegression()                    # 로지스틱 회귀

    # 모델별 하이퍼 파라미터 튜닝을 위한 후보값 그리드 정의
    
    # 1) Decision Tree 파라미터
    dt_params = {
        'max_depth': [2, 3, 5, 10],       # 트리의 최대 깊이 제한
        'min_samples_split': [2, 3, 5],   # 노드를 분할하기 위한 최소 샘플 수
        'min_samples_leaf': [1, 5, 8]     # 리프 노드가 되기 위한 최소 샘플 수
    }
    
    # 2) Random Forest 파라미터
    rf_params = {
        'n_estimators': [50, 100],        # 생성할 트리의 개수
        'max_depth': [6, 8, 10, 12],      # 트리의 최대 깊이
        'min_samples_leaf': [8, 12, 18],  # 리프 노드 최소 샘플 수
        'min_samples_split': [8, 16, 20]  # 노드 분할 최소 샘플 수
    }
    
    # 3) Logistic Regression 파라미터
    lr_params = {
        'penalty': ['l2', 'l1'],  # 규제(Regularization) 유형 (l2: 릿지, l1: 라쏘)
        'C': [0.01, 0.1, 1, 10]   # 규제 강도 (작을수록 규제가 강함) - 과적합 방지용
    }

    # GridSearchCV 수행을 위해 (모델이름, 모델객체, 파라미터) 튜플 리스트 생성
    models_to_tune = [
        ('DecisionTree', dt_clf, dt_params),
        ('RandomForest', rf_clf, rf_params),
        ('LogisticRegression', lr_clf, lr_params)
    ]

    # 전체 모델 중 가장 좋은 모델을 찾기 위한 변수 초기화
    best_overall_model = None
    best_overall_score = 0
    best_overall_name = ""

    print("\n### 모든 모델에 대한 GridSearchCV 수행 ###")

    # 반복문을 통해 각 모델별로 GridSearchCV를 수행합니다.
    for name, model, params in models_to_tune:
        print(f"\n--- {name} Tuning ---")
        
        # LogisticRegression의 경우 l1 규제를 사용하려면 solver 설정이 필요합니다.
        if name == 'LogisticRegression':
             # 'liblinear' solver는 l1, l2 규제를 모두 지원하며 작은 데이터셋에 적합합니다.
            model.solver = 'liblinear' 

        # GridSearchCV 객체 생성
        # scoring='accuracy': 정확도를 기준으로 평가
        # cv=5: 5-Fold 교차 검증 수행
        # n_jobs=-1: 병렬 처리로 모든 CPU 코어를 사용하여 속도 향상
        grid_cv = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1)
        
        # 학습 데이터로 그리드 서치 수행 (최적 파라미터 탐색)
        grid_cv.fit(X_train, y_train)

        # 찾은 최적 파라미터와 교차 검증 평균 정확도 출력
        print(f'{name} 최적 파라미터: {grid_cv.best_params_}')
        # grid_cv.best_score_는 교차 검증 단계에서의 평균 정확도입니다.
        print(f'{name} 최고 교차 검증 정확도: {grid_cv.best_score_:.4f}')

        # 최적의 파라미터로 학습된 모델(best_estimator_)을 가져와 테스트 데이터를 평가합니다.
        best_estimator = grid_cv.best_estimator_
        predictions = best_estimator.predict(X_test)
        
        # 테스트 세트 정확도 계산
        accuracy = accuracy_score(y_test, predictions)
        print(f'{name} 테스트 세트 정확도: {accuracy:.4f}')

        # 현재 튜닝된 모델이 지금까지 확인된 최고 모델보다 성능이 좋다면 정보를 갱신합니다.
        if accuracy > best_overall_score:
            best_overall_score = accuracy
            best_overall_model = best_estimator
            best_overall_name = name

    # 모든 모델에 대한 튜닝 및 평가가 끝난 후 최종 결과 출력
    print("\n=============================================")
    print(f"최종 결론: 가장 성능이 좋은 모델은 '{best_overall_name}' 입니다.")
    print(f"최고 정확도: {best_overall_score:.4f}")
    
    # 최고 모델의 주요 파라미터 정보 출력 (모델 종류에 따라 다름)
    if best_overall_name == 'DecisionTree':
        print(f"최적 파라미터: {best_overall_model.get_params()['max_depth']} (depth) 등")
    elif best_overall_name == 'RandomForest':
        print(f"트리 개수: {best_overall_model.get_params()['n_estimators']}")
    print("=============================================")

if __name__ == "__main__":
    main()
