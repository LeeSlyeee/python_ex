# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석과 머신러닝 모델링에 필요한 핵심 라이브러리들을 불러옵니다.
import numpy as np                  # 수치 계산(배열, 행렬 등)을 효율적으로 처리하기 위한 라이브러리
import pandas as pd                 # 데이터프레임(표 형태)을 다루고 분석하기 위한 필수 라이브러리
from sklearn.model_selection import train_test_split # 학습(Train) 데이터와 검증(Validation/Test) 데이터를 나누는 함수
from sklearn.impute import SimpleImputer # 결측치(NaN)를 특정 전략(평균, 중앙값 등)으로 채워주는 클래스
from sklearn.preprocessing import LabelEncoder # 범주형(문자열) 데이터를 수치형(정수)으로 변환하는 클래스 (0, 1, 2...)
from sklearn.preprocessing import MinMaxScaler # 데이터를 0과 1 사이의 범위로 스케일링(정규화)하는 클래스
from sklearn.preprocessing import StandardScaler # 데이터를 평균 0, 표준편차 1인 분포로 스케일링(표준화)하는 클래스


# =============================================================================
# [2] 데이터 로드 및 확인 (Data Loading & Inspection)
# =============================================================================
# 타이타닉 생존자 데이터를 웹에서 로드합니다.
train = pd.read_csv("http://bit.ly/fc-ml-titanic")

# 데이터의 앞 5행을 출력하여 구조를 파악합니다.
print("--- [Head] 데이터 미리보기 ---")
print(train.head())

# 모델 학습에 사용할 특성(Features)과 예측할 타겟(Label)을 지정합니다.
features = ['Pclass', 'Sex', 'Age', 'Fare'] # 학습에 사용할 컬럼들 (좌석등급, 성별, 나이, 요금)
label = ['Survived']                        # 예측 목표 컬럼 (생존 여부: 0=사망, 1=생존)

# 특성 데이터와 라벨 데이터의 일부를 확인합니다.
print("\n--- [Features] 입력 데이터 ---")
print(train[features].head())
print("\n--- [Label] 정답 데이터 ---")
print(train[label].head())


# =============================================================================
# [3] 데이터 분할 (Data Split)
# =============================================================================
# 전체 데이터를 학습용(Train)과 검증용(Validation)으로 나눕니다.
# test_size=0.2 : 전체의 20%를 검증용으로 사용 (80%는 학습용)
# shuffle=True  : 데이터를 순서대로 자르지 않고 무작위로 섞어서 나눔 (데이터 편향 방지)
# random_state=30 : 실행할 때마다 같은 결과가 나오도록 난수 시드 고정
x_train, x_valid, y_train, y_valid = train_test_split(train[features], train[label], test_size=0.2, shuffle=True, random_state=30)

print("\n--- [Shape] 데이터 분할 결과 형태 ---")
print(f"학습 데이터: {x_train.shape}, 정답: {y_train.shape}")
print(f"검증 데이터: {x_valid.shape}, 정답: {y_valid.shape}")


# =============================================================================
# [4] 결측치 확인 (Checking Missing Values)
# =============================================================================
print("\n--- [Info] 데이터 정보 ---")
print(train.info()) # 컬럼별 데이터 타입과 Non-Null 개수 확인

print("\n--- [NULL] 결측치 총 개수 ---")
print(train.isnull().sum()) # 각 컬럼별 NaN 개수 출력

print("\n--- Age 컬럼 결측치 ---")
print(train['Age'].isnull().sum()) # 'Age' 컬럼만 따로 확인


# =============================================================================
# [5] 결측치 처리 (Handling Missing Values)
# =============================================================================

# 5-1. Pandas fillna() 사용
# -----------------------------------------------------------------------------
# NaN을 0으로 채웠을 때의 통계 정보
print("\n--- [Age] 0으로 채움 ---")
print(train['Age'].fillna(0).describe())

# NaN을 전체 나이의 평균값으로 채웠을 때의 통계 정보
print("\n--- [Age] 평균값으로 채움 ---")
print(train['Age'].fillna(train['Age'].mean()).describe())


# 5-2. Scikit-learn SimpleImputer 사용 (평균 대체)
# -----------------------------------------------------------------------------
# strategy='mean' : 결측치를 해당 컬럼의 평균값으로 대체
imputer = SimpleImputer(strategy='mean')

# fit(): 데이터의 평균을 계산하는 과정
imputer.fit(train[['Age', 'Pclass']]) 

# transform(): 계산된 평균값으로 실제 결측치를 채우는 과정
result = imputer.transform(train[['Age', 'Pclass']])
print("\n--- [Imputer] 변환 결과 (일부) ---")
print(result[:5]) # 결과는 numpy array 형태

# 원본 데이터프레임에 적용
train[['Age', 'Pclass']] = result
print("\n--- [Result] Imputer 적용 후 결측치 확인 ---")
print(train[['Age', 'Pclass']].isnull().sum())
print(train[['Age', 'Pclass']].describe())


# [실습 초기화] 다른 실습을 위해 데이터 다시 로드
train = pd.read_csv('https://bit.ly/fc-ml-titanic')

# 5-3. Scikit-learn SimpleImputer 사용 (중앙값 대체)
# -----------------------------------------------------------------------------
# strategy='median' : 결측치를 중앙값으로 대체 (이상치에 덜 민감함)
imputer = SimpleImputer(strategy='median')

# fit_transform(): fit과 transform을 한 번에 수행
result = imputer.fit_transform(train[['Age', 'Pclass']])
train[['Age', 'Pclass']] = result

print("\n--- [Median Impute] 중앙값 대체 결과 ---")
print(train[['Age', 'Pclass']].isnull().sum())
print(train[['Age', 'Pclass']].describe())


# [실습 초기화] 데이터 다시 로드
train = pd.read_csv('https://bit.ly/fc-ml-titanic')
train_copy = train.copy()

# 5-4. 범주형 데이터(Categorical) 결측치 처리
# -----------------------------------------------------------------------------
# 'Embarked' (탑승 항구) 결측치를 문자열 'S'로 채우기
print("\n--- [Embarked] 'S'로 단순 채우기 ---")
print(train['Embarked'].fillna('S').head())

# strategy='most_frequent' : 최빈값(가장 자주 등장하는 값)으로 대체
imputer = SimpleImputer(strategy='most_frequent')
result = imputer.fit_transform(train[['Embarked', 'Cabin']])
train[['Embarked', 'Cabin']] = result

print("\n--- [Most Frequent] 최빈값 대체 결과 ---")
print(train[['Embarked', 'Cabin']].isnull().sum())


# =============================================================================
# [6] 데이터 인코딩 (Data Encoding) - 문자를 숫자로 변환
# =============================================================================
train.info()

# 6-1. 사용자 정의 함수 (apply 사용)
# -----------------------------------------------------------------------------
def convert(data):
    if data == 'male':
        return 1
    elif data == 'female':
        return 0

print("\n--- [Custom Encoding] 사용자 정의 함수 적용 ---")
print(train['Sex'].value_counts())
print(train['Sex'].apply(convert).head())

# 6-2. LabelEncoder 사용 (Scikit-learn)
# -----------------------------------------------------------------------------
# 범주형 데이터를 0, 1, 2... 정수로 자동 변환
le = LabelEncoder()
train['Sex_num'] = le.fit_transform(train['Sex'])

print("\n--- [LabelEncoder] 성별 변환 결과 ---")
print(train['Sex_num'].value_counts())
print(f"Classes: {le.classes_}") # ['female', 'male'] -> 0, 1 순서 확인

# inverse_transform: 숫자를 다시 문자로 복원
print(f"복원 테스트: {le.inverse_transform([0, 1, 1, 0])}")


# [Tip] 결측치가 있는 상태에서 LabelEncoder 사용 시 주의사항
# -----------------------------------------------------------------------------
print("\n--- [Check] 결측치가 포함된 데이터 인코딩 ---")
print(f"원본 결측치 수: {train_copy['Embarked'].isna().sum()}")
# 결측치가 있으면 에러가 나거나, 결측치도 하나의 범주로 처리됨 (버전에 따라 다름/보통 문자열로 변환 필요)
# 여기서는 NaN 값도 하나의 unique 값으로 취급되어 인코딩 됨
train_copy["le_Embarked"] = le.fit_transform(train_copy['Embarked'])
print(f"인코딩된 유니크 값: {train_copy['le_Embarked'].unique()}")

# 결측치를 먼저 채우고 인코딩하는 것이 정석
train['Embarked'] = train['Embarked'].fillna('S')
train["le_Embarked"] = le.fit_transform(train['Embarked'])
print(f"결측치 처리 후 인코딩 결과: {train['le_Embarked'].unique()}")


# [실습 초기화]
train = pd.read_csv('https://bit.ly/fc-ml-titanic')
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked_num'] = LabelEncoder().fit_transform(train['Embarked'])

# 6-3. One-Hot Encoding (원-핫 인코딩)
# -----------------------------------------------------------------------------
# 0, 1, 2 형태의 라벨 인코딩은 숫자의 크기가 모델에 잘못된 영향을 줄 수 있음 (예: 2가 0보다 크다?)
# 이를 방지하기 위해 해당 카테고리만 1, 나머지는 0으로 만드는 방식
print("\n--- [One-Hot Encoding] ---")
print(train['Embarked_num'][:6])

# pd.get_dummies() : 원-핫 인코딩 수행
one_hot = pd.get_dummies(train['Embarked_num'][:6])
one_hot.columns = ['C','Q','S'] # 컬럼 이름 보기 좋게 변경
print(one_hot.astype(int)) # True/False -> 1/0 변환 출력

# Embarked는 탑승 항구의 이니셜을 나타냈습니다.
# 그런데, 우리는 이전에 배운 내용처럼 LabelEncoder 를 통해서 수치형으로 변환해주었습니다.
# 하지만, 이대로 데이터를 기계학습을 시키면, 기계는 데이터 안에서 관계를 학습합니다.
# 즉, 'S' = 2, 'Q' = 1 이라고 되어 있는데, Q + Q = S 가 된다 라고 학습해버린다는 것이죠.
# 그렇기 때문에, 독립적인 데이터는 별도의 column으로 분리하고, 각각의 컬럼에 해당 값에만 True 나머지는 False를 갖습니다. 우리는 이것을 원 핫 인코딩 한다고 합니다.
# column 을 분리시켜 카테고리형 -> 수치형으로 변환하면서 생기는 수치형 값 의 관계를 끊어주어서 독립적인 형태로 바꾸어 줍니다.
# 원핫인코딩은 카테고리 (계절, 항구, 성별, 종류, 한식/일식/중식...)의 특성을 가지는 column에 대해서 적용 합니다.


# =============================================================================
# [7] 데이터 스케일링 (Data Scaling) - 데이터 범위 조정
# =============================================================================

# 7-1. MinMaxScaler (정규화)
# -----------------------------------------------------------------------------
# 데이터의 최솟값을 0, 최댓값을 1로 맞춤. 이상치에 민감함.
movie = {'naver': [2, 4, 6, 8, 10], 'netflix': [1, 2, 3, 4, 5]}
movie_df = pd.DataFrame(data=movie)
print("\n--- [MinMax Scaling] 전 ---")
print(movie_df)

min_max_scaler = MinMaxScaler()
min_max_movie = min_max_scaler.fit_transform(movie_df)

print("--- [MinMax Scaling] 후 (0~1 범위) ---")
print(pd.DataFrame(min_max_movie, columns=['naver', 'netflix']))


# 7-2. StandardScaler (표준화)
# -----------------------------------------------------------------------------
# 평균을 0, 표준편차를 1로 맞춤 (표준정규분포). 이상치의 영향을 덜 받음.
standard_scaler = StandardScaler()
x = np.arange(10)
x[9] = 1000 # 마지막에 매우 큰 이상치 추가

print("\n--- [Standard Scaling] ---")
print(f"원본 - 평균: {x.mean()}, 표준편차: {x.std()}")

# 1차원 배열을 2차원(-1, 1)으로 바꿔서 넣어줘야 함
scaled = standard_scaler.fit_transform(x.reshape(-1, 1))

print(f"스케일링 후 - 평균: {scaled.mean():.2f}, 표준편차: {scaled.std():.2f}")
print("평균이 0, 표준편차가 1에 근사하게 변환됨")

