from sklearn.preprocessing import LabelEncoder  # 카테고리형 피처를 숫자형으로 변환 (Label Encoding)
from sklearn.preprocessing import OneHotEncoder  # 카테고리형 피처를 원-핫 인코딩으로 변환
from sklearn.preprocessing import StandardScaler  # 평균 0, 분산 1로 변환 (표준화)
from sklearn.preprocessing import MinMaxScaler  # 0과 1 사이의 값으로 변환 (정규화)
from sklearn.datasets import load_iris  # 붓꽃 데이터셋 로드
import numpy as np  # 수치 연산
import pandas as pd  # 데이터프레임 처리

# ==========================================
# 1. 레이블 인코딩 (Label Encoding)
# ==========================================
# 문자열(String)로 된 카테고리형 데이터를 숫자(Number)로 변환합니다.
# TV -> 0, 냉장고 -> 1, ... 이런 식으로 매핑됩니다.

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# LabelEncoder 객체 생성
encoder = LabelEncoder()
# 1. fit(): 데이터의 고유한 값(Unique Value)을 찾아 정렬하고 인코딩 규칙을 학습합니다.
encoder.fit(items)

# 2. transform(): 학습된 규칙에 따라 실제 데이터를 숫자로 변환합니다.
labels = encoder.transform(items)
print(f'인코딩 변환값: {labels}')  # [0 1 4 5 3 3 2 2] 등 정렬 순서대로 0부터 번호 부여

# classes_ 속성에는 인코딩된 문자열 원본이 정렬된 순서대로 들어있습니다. (0번부터 순서대로)
print(f'인코딩 클래스: {encoder.classes_}')

# inverse_transform(): 인코딩된 숫자를 다시 원본 문자열로 복구(디코딩)합니다.
print(f'디코딩 원본 값: {encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3])}')


# ==========================================
# 2. 원-핫 인코딩 (One-Hot Encoding)
# ==========================================
# 레이블 인코딩의 문제점(숫자 크기가 의미를 가짐 -> 1등, 2등 처럼 인식)을 해결하기 위해 사용합니다.
# 고유한 값의 개수만큼 컬럼을 만들고, 해당되는 컬럼에만 1, 나머지는 0을 표시합니다. (희소 행렬 형태)

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# 먼저 숫자값으로 변환을 위해 LabelEncoder 사용
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

# 2차원 데이터로 변환해야 OneHotEncoder에 넣을 수 있습니다. (-1은 '나머지 차원은 알아서 맞춰라'는 뜻)
labels = labels.reshape(-1, 1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)

# toarray()를 사용해 희소 행렬(Sparse Matrix)을 밀집 행렬(Dense Matrix)로 변환해 눈으로 확인합니다.
print(f'원-핫 인코딩 변환값: \n{oh_labels.toarray()}')
print(f'원-핫 인코딩 데이터 차원: {oh_labels.shape}') 


# Pandas의 get_dummies()를 사용하면 훨씬 간편하게 원-핫 인코딩을 할 수 있습니다.
# (별도의 숫자 변환 과정 없이 문자열 데이터를 바로 처리해 줍니다.)
df = pd.DataFrame(items, columns=['item']) # 주의: 컬럼명 딕셔너리가 아닌 리스트로 수정
print('\n[Pandas get_dummies 결과]')
print(pd.get_dummies(df))


# ==========================================
# 3. 피처 스케일링 (Feature Scaling) - StandardScaler
# ==========================================
# 데이터를 평균 0, 분산 1인 정규 분포(가우시안 분포) 형태로 변환합니다.
# SVM, 선형 회귀, 로지스틱 회귀 등에서 성능 향상에 도움이 됩니다.

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)

print('\n[StandardScaler 적용 전]')
print(f'feature 들의 평균 값: \n{iris_df.mean()}')
print(f'feature 들의 분산 값: \n{iris_df.var()}')


scaler = StandardScaler()
# fit()으로 데이터의 평균과 분산을 계산하고, transform()으로 변환합니다.
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform 시 반환값은 ndarray이므로 보기 좋게 DataFrame으로 변환
# iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)

print('\n[StandardScaler 적용 후]')
print(f'feature 들의 평균 값: \n{iris_scaled.mean()}') # 0에 매우 가까운 값
print(f'feature 들의 분산 값: \n{iris_scaled.var()}')  # 1에 매우 가까운 값


# ==========================================
# 4. 피처 스케일링 (Feature Scaling) - MinMaxScaler
# ==========================================
# 데이터 값을 0과 1 사이의 범위로 변환합니다. (음수가 있으면 -1 ~ 1)
# 데이터 분포가 정규 분포가 아닐 때 유용합니다.

scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

print('\n[MinMaxScaler 적용 후]')
print(f'feature 들의 최소 값: {iris_scaled.min()}') # 0.0
print(f'feature 들의 최대 값: {iris_scaled.max()}') # 1.0


# ==========================================
# 5. 스케일링 시 주의사항 (Data Leakage 방지)
# ==========================================
# 학습 데이터(Train)와 테스트 데이터(Test)를 스케일링할 때 가장 중요한 원칙:
# "학습 데이터의 기준(fit)을 테스트 데이터에도 똑같이 적용해야 한다!"

train_array = np.arange(0, 11).reshape(-1, 1) # 0부터 10까지
test_array = np.arange(0, 6).reshape(-1, 1)  # 0부터 5까지

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# 1. 학습 데이터로 fit(): 학습 데이터의 최솟값(0), 최댓값(10)을 찾습니다.
scaler.fit(train_array)
# 2. 학습 데이터 transform(): 0~10을 0~1로 변환합니다. (10 -> 1.0)
train_scaled = scaler.transform(train_array)

print(f'\n원본 train_array 데이터: {np.round(train_array.reshape(-1), 2)}')
print(f'Scale된 train_array 데이터: {np.round(train_scaled.reshape(-1), 2)}')


# [잘못된 예시] 테스트 데이터로 다시 fit()을 수행하는 경우
# 테스트 데이터(0~5)를 기준으로 새로운 척도를 만들어버립니다. (5 -> 1.0)
# 이렇게 되면 학습 데이터의 1.0(100점)과 테스트 데이터의 1.0(50점)이 서로 다른 의미를 가지게 되어 모델이 혼란스러워합니다.
scaler.fit(test_array)
test_scaled = scaler.transform(test_array)

print(f'\n[잘못된 예] 원본 test_array 데이터: {np.round(test_array.reshape(-1), 2)}')
print(f'[잘못된 예] Scale된 test_array 데이터: {np.round(test_scaled.reshape(-1), 2)}')


# [올바른 예시] 학습 데이터 때 fit()한 scaler 객체를 그대로 사용하여 transform()만 수행해야 합니다.
scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)

print(f'\n[올바른 예] 원본 train_array 데이터: {np.round(train_array.reshape(-1), 2)}')
print(f'[올바른 예] Scale된 train_array 데이터: {np.round(train_scaled.reshape(-1), 2)}')

# 테스트 데이터는 절대 fit() 하지 않고 바로 transform() 만 합니다!
# 그래야 "학습 데이터의 10점이 1.0이니까, 테스트 데이터의 5점은 0.5구나" 하고 올바르게 번역됩니다.
test_scaled = scaler.transform(test_array)

print(f'\n[올바른 예] 원본 test_array 데이터: {np.round(test_array.reshape(-1), 2)}')
print(f'[올바른 예] Scale된 test_array 데이터: {np.round(test_scaled.reshape(-1), 2)}')
