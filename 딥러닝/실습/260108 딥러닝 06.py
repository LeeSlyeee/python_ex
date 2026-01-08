# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석, 시각화, 딥러닝 모델링에 필요한 라이브러리를 불러옵니다.

import warnings
warnings.filterwarnings('ignore') # 실행 시 불필요한 경고 메시지를 숨깁니다.

# TensorFlow Keras: 딥러닝 모델 구축
from tensorflow.keras.models import Sequential # 순차적 모델 생성 클래스
from tensorflow.keras.layers import Dense      # 완전 연결층
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # 모델 최적화를 위한 콜백

# Scikit-learn: 데이터 분리
from sklearn.model_selection import train_test_split # 학습/테스트 데이터셋 분리

# 데이터 처리 및 시각화
import os                       # 시스템 경로 제어
import pandas as pd             # 데이터 처리 (DataFrame)
import numpy as np              # 수치 연산
import matplotlib.pyplot as plt # 기본 그래프 시각화
import seaborn as sns           # 고급 통계 시각화


# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================

# 현재 스크립트 파일 위치를 기준으로 데이터 파일 경로 설정
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../data/house_train.csv")

# CSV 파일 로드 (헤더가 있는 일반적인 CSV 파일)
df = pd.read_csv(file_path)

# 데이터프레임 구조 확인
print(df)

# 데이터 타입 확인 (숫자형 vs 문자형 등)
print(df.dtypes)


# =============================================================================
# [3] 데이터 탐색 및 전처리 (Data Exploration & Preprocessing)
# =============================================================================

# 결측치(Null) 확인
# 각 컬럼별 결측치 개수를 내림차순으로 정렬하여 상위 20개 출력
print(df.isnull().sum().sort_values(ascending=False).head(20))

# 카테고리형(문자열) 변수 원-핫 인코딩 (One-Hot Encoding)
# 문자열 데이터를 0과 1의 숫자 데이터로 변환합니다.
df = pd.get_dummies(df)

# 결측치 처리
# 결측치가 있는 곳을 각 컬럼의 평균값(mean)으로 채웁니다.
df = df.fillna(df.mean())

# 저리된 데이터 확인
print(df)


# 상관관계 분석 (Correlation Analysis)
# 각 변수들 간의 상관계수 구하기
df_corr = df.corr()

# 'SalePrice'(집값)와 상관관계가 높은 순서대로 정렬
df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)

# 집값과 가장 상관관계가 높은 상위 10개 변수 출력
print(df_corr_sort['SalePrice'].head(10))


# 주요 변수(상관관계가 높은 변수들) 간의 관계 시각화 (Pairplot)
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
sns.pairplot(df[cols])
plt.show()


# =============================================================================
# [4] 학습/테스트 데이터 분리 (Train/Test Split)
# =============================================================================

# 학습에 사용할 속성(Feature) 선택
# 상관관계가 높았던 주요 변수들을 선택하여 학습 데이터로 사용합니다.
cols_train = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
X_train_pre = df[cols_train] # Feature 데이터

# 타겟(Target) 데이터 설정 (예측할 집값)
y = df['SalePrice'].values

# 전체 데이터를 학습용(80%)과 테스트용(20%)으로 분리
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)

# 분리된 데이터 확인
print(X_train)
print(X_train.shape[1]) # 입력 속성의 개수 (Input Dimension 확인)
print(X_train.shape[0]) # 학습 데이터 샘플 수


# =============================================================================
# [5] 모델 설계 및 학습 (Model Design & Training)
# =============================================================================

# 모델 구조 설계
model = Sequential()
# 은닉층 1: 10개 노드, 입력 차원은 X_train의 컬럼 개수만큼 설정
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
# 은닉층 2: 30개 노드
model.add(Dense(30, activation='relu'))
# 은닉층 3: 40개 노드
model.add(Dense(40, activation='relu'))
# 출력층: 1개 노드 (집값이라는 연속된 수치를 예측하는 회귀 문제이므로 활성화 함수 없이 출력)
model.add(Dense(1))

# 모델 요약 정보 출력
model.summary()

# 모델 컴파일
# loss='mean_squared_error': 회귀 문제에서의 오차 함수 (평균 제곱 오차)
# optimizer='adam': 최적화 알고리즘
model.compile(optimizer ='adam', loss = 'mean_squared_error')

# 조기 종료(EarlyStopping) 설정
# val_loss가 20번 이상 개선되지 않으면 학습 조기 중단
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 모델 저장 경로 설정
modelpath = os.path.join(script_dir, '../data/model/house.keras')

# 체크포인트(ModelCheckpoint) 설정
# 최적의 모델(val_loss 기준)을 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 모델 학습
# 학습 데이터의 25%를 검증셋으로 사용 (validation_split=0.25)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])


# =============================================================================
# [6] 예측 및 결과 시각화 (Prediction & Visualization)
# =============================================================================

# 테스트 데이터 출력 (확인용)
print(X_test)

# 모델을 사용하여 테스트 데이터의 집값 예측
# 상위 10개 예측값 출력
print(model.predict(X_test)[:10])

# 1차원 배열로 변환하여 보기 쉽게 출력
print(model.predict(X_test).flatten()[:10])


# 실제값과 예측값 비교
real_prices = [] # 실제 가격 저장 리스트
pred_prices = [] # 예측 가격 저장 리스트
X_num = []       # 그래프 X축 좌표 (샘플 번호)

n_iter = 0
Y_prediction = model.predict(X_test).flatten() # 전체 테스트 데이터 예측

# 25개의 샘플에 대해 실제값 vs 예측값 비교 출력
for i in range(25):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)

# 결과 그래프 그리기
# 파란색: 예측 가격, 주황색(또는 다른색): 실제 가격
plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend() # 범례 표시
plt.show()   # 그래프 출력