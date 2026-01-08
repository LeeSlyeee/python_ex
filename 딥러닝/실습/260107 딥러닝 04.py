# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석, 파일 제어 및 딥러닝 모델링에 필요한 핵심 라이브러리들을 불러옵니다.

import warnings
warnings.filterwarnings('ignore') # 실행 시 불필요한 경고 메시지(FutureWarning 등)를 숨깁니다.

import pandas as pd             # 데이터 처리 및 분석 (DataFrame 활용)
import os                       # 시스템 경로 및 파일 제어 (파일 저장/로드 경로 설정 등)

# TensorFlow Keras: 딥러닝 모델 구축을 위한 고수준 API
from tensorflow.keras.models import Sequential, load_model # 순차적 모델 생성 및 저장된 모델 불러오기
from tensorflow.keras.layers import Dense      # 완전 연결층 (Fully Connected Layer)

# Scikit-learn: 머신러닝 데이터 전처리 및 평가 도구
from sklearn.model_selection import train_test_split # 학습/테스트 데이터셋 분리 함수
from sklearn.model_selection import KFold      # K-겹 교차 검증 (K-Fold Cross Validation)


# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================

# 현재 스크립트 파일의 경로를 기준으로 데이터 파일 경로 설정
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../data/sonar3.csv")

# 헤더가 없는 CSV 파일 로드 (첫 번째 행도 데이터로 인식)
df = pd.read_csv(file_path, header=None)

# 데이터의 상위 5개 행 출력 (데이터 구조 및 값 확인)
print(df.head())


# =============================================================================
# [3] 데이터 탐색 및 전처리 (Data Exploration & Preprocessing)
# =============================================================================

# 타겟(마지막 컬럼, 인덱스 60)의 클래스별 데이터 개수 확인
# 예: 광물(M) vs 암석(R)의 분포를 확인하여 데이터 불균형 여부 파악
print(df[60].value_counts())

# Feature(X)와 Target(y) 분리
# X: 0번부터 59번 컬럼까지의 속성 데이터 (총 60개 속성)
X = df.iloc[:,0:60]
# y: 60번 컬럼, 예측해야 할 정답 클래스 (타겟)
y = df.iloc[:,60]


# =============================================================================
# [4] 모델 설계 및 학습 (Model Design & Training - Full Data)
# =============================================================================
# 전체 데이터를 사용하여 모델을 학습시키는 기본 과정입니다.

# 순차적 모델 객체 생성 (Layer를 선형으로 쌓는 스택)
model = Sequential()

# 첫 번째 은닉층 (Hidden Layer 1)
# 24: 뉴런(노드) 개수
# input_dim=60: 입력 속성이 60개이므로 입력층의 노드 수는 60
# activation='relu': 은닉층의 활성화 함수로 ReLU 사용
model.add(Dense(24, input_dim=60, activation='relu'))

# 두 번째 은닉층 (Hidden Layer 2)
# 10: 뉴런 개수, 활성화 함수로 ReLU 사용
model.add(Dense(10, activation='relu'))

# 세 번째 은닉층 (Hidden Layer 3) - 예제로 추가된 층
# 10: 뉴런 개수, activation='tanh': 활성화 함수로 Hyperbolic Tangent 사용
model.add(Dense(10, activation='tanh'))

# 출력층 (Output Layer)
# 1: 이진 분류이므로 출력 뉴런은 1개
# activation='sigmoid': 0과 1 사이의 확률값을 반환 (0.5 기준 분류)
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일 (Compile)
# loss='binary_crossentropy': 이진 분류를 위한 오차 함수
# optimizer='adam': 효율적인 경사 하강법 최적화 알고리즘
# metrics=['accuracy']: 학습 도중 정확도를 모니터링 지표로 사용
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습 (Training)
# epochs=200: 전체 데이터셋을 200번 반복 학습
# batch_size=10: 10개의 샘플마다 가중치를 갱신
history=model.fit(X, y, epochs=200, batch_size=10)


# =============================================================================
# [5] 학습/테스트 데이터 분리 및 평가 (Train/Test Split & Evaluation)
# =============================================================================
# 모델의 일반화 성능을 평가하기 위해 데이터를 학습용과 테스트용으로 분리하여 실습합니다.

# 데이터 다시 로드 (앞선 실습과 독립성을 위해 초기화)
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,0:60]
y = df.iloc[:,60]

# 학습 데이터(70%)와 테스트 데이터(30%)로 분리
# test_size=0.3: 전체의 30%를 테스트 셋으로 할당
# shuffle=True: 데이터를 무작위로 섞어서 분할 (편향 방지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# 새로운 모델 설계 (위와 동일한 구조)
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 데이터(X_train, y_train)로만 모델 학습 수행
history=model.fit(X_train, y_train, epochs=200, batch_size=10)

# 테스트 데이터(X_test, y_test)로 모델 성능 평가
# score 리스트 반환: [Loss, Accuracy]
score=model.evaluate(X_test, y_test)

# 테스트 정확도 출력
print('Test accuracy:', score[1])


# =============================================================================
# [6] 모델 저장 및 불러오기 (Model Save & Load)
# =============================================================================
# 학습된 모델을 파일로 저장하여 나중에 재사용하는 방법입니다.

# 모델 저장 경로 설정
# Keras 3.0부터는 `.keras` 확장자 사용을 권장합니다.
model_path = os.path.join(script_dir, '../data/model/my_model.keras')

# 저장할 디렉토리가 없으면 생성 (안전한 파일 저장을 위해)
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))

# 현재 학습된 모델을 파일로 저장
model.save(model_path)


# 메모리에서 모델 모델 객체 삭제 (불러오기 기능을 확실히 테스트하기 위함)
del model


# 저장된 모델 파일 불러오기
# load_model 함수를 사용하여 구조와 가중치가 포함된 모델을 복원합니다.
model = load_model(model_path)

# 불러온 모델로 테스트 데이터 평가 수행 (저장 전과 결과가 동일해야 함)
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])


# =============================================================================
# [7] K-겹 교차 검증 (K-Fold Cross Validation)
# =============================================================================
# 데이터가 적을 때 과적합을 방지하고 모델의 신뢰성을 높이기 위한 검증 방법입니다.

# 데이터 다시 로드
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,0:60]
y = df.iloc[:,60]


# 교차 검증 설정 (5-Fold)
# n_splits=5: 데이터를 5개의 그룹(폴드)으로 나눕니다.
k=5
kfold = KFold(n_splits=k, shuffle=True)

# 각 폴드에서의 정확도를 저장할 리스트 생성
acc_score = []


# 모델 생성 함수 정의
# 교차 검증 시 매번 새로운(초기화된) 모델을 생성해야 하므로 함수로 만듭니다.
def model_fn():
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# K-겹 교차 검증 수행
# kfold.split(X): 데이터를 학습용 인덱스와 테스트용 인덱스로 나누어 반환 (총 k번 반복)
for train_index , test_index in kfold.split(X):
    # 인덱스를 사용하여 실제 학습 데이터와 테스트 데이터 추출
    X_train , X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train , y_test = y.iloc[train_index], y.iloc[test_index]

    # 매 반복마다 새로운 모델 생성
    model = model_fn()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 모델 학습 (verbose=0: 학습 과정 로그를 출력하지 않음)
    history=model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)

    # 해당 폴드의 테스트 데이터로 정확도 평가
    accuracy = model.evaluate(X_test, y_test)[1] 
    
    # 정확도 리스트에 추가
    acc_score.append(accuracy)

# k번 실시된 정확도의 평균 계산
avg_acc_score = sum(acc_score)/k

# 각 폴드별 정확도와 전체 평균 정확도 출력
print('정확도:', acc_score)
print('정확도 평균:', avg_acc_score)