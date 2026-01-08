# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
import warnings
warnings.filterwarnings('ignore') # 불필요한 경고 메시지 숨기기

import pandas as pd             # 데이터 처리 및 분석
import os                       # 시스템 경로 및 파일 제어

# Deep Learning Framework (Keras)
from tensorflow.keras.models import Sequential, load_model # 모델 구조 및 저장된 모델 로드
from tensorflow.keras.layers import Dense      # 완전 연결층
from sklearn.model_selection import train_test_split # 학습/테스트 데이터 분리
from sklearn.model_selection import KFold      # K-겹 교차 검증


# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================

# 현재 스크립트 위치 기반으로 데이터 파일 경로 설정
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../data/sonar3.csv")

# 헤더가 없는 CSV 파일 로드
df = pd.read_csv(file_path, header=None)

# 데이터 상위 5행 출력 확인
print(df.head())


# =============================================================================
# [3] 데이터 탐색 및 전처리 (Data Exploration & Preprocessing)
# =============================================================================

# 마지막 컬럼(60번 인덱스)의 값 분포 확인 (광물 vs 암석)
print(df[60].value_counts())

# Feature(X)와 Target(y) 분리
X = df.iloc[:,0:60] # 0~59번 컬럼: 속성 데이터
y = df.iloc[:,60]   # 60번 컬럼: 클래스 (타겟)


# =============================================================================
# [4] 모델 설계 및 학습 (Model Design & Training - Full Data)
# =============================================================================

# 순차적 모델 객체 생성
model = Sequential()

# 은닉층 및 출력층 추가
model.add(Dense(24, input_dim=60, activation='relu')) # 은닉층 1
model.add(Dense(10, activation='relu'))               # 은닉층 2
model.add(Dense(10, activation='tanh'))               # 은닉층 3 (tanh 활성화 함수)
model.add(Dense(1, activation='sigmoid'))             # 출력층 (이진 분류)

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습 (전체 데이터 사용)
history=model.fit(X, y, epochs=200, batch_size=10)


# =============================================================================
# [5] 학습/테스트 데이터 분리 및 평가 (Train/Test Split & Evaluation)
# =============================================================================

# 데이터 다시 로드 (실습 진행을 위해 초기화)
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,0:60]
y = df.iloc[:,60]

# 학습 데이터(70%)와 테스트 데이터(30%)로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# 새로운 모델 설계
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 데이터로 모델 학습
history=model.fit(X_train, y_train, epochs=200, batch_size=10)

# 테스트 데이터로 모델 평가
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])


# =============================================================================
# [6] 모델 저장 및 불러오기 (Model Save & Load)
# =============================================================================

# 모델 저장 경로 설정 (Keras 3.0부터는 .keras 확장자를 권장합니다)
model_path = os.path.join(script_dir, '../data/model/my_model.keras')

# 저장할 디렉토리가 없으면 생성
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))

# 모델 저장
model.save(model_path)


# 메모리에서 모델 삭제 (불러오기 테스트용)
del model


# 저장된 모델 불러오기
model = load_model(model_path)

# 불러온 모델로 평가 수행
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])


# =============================================================================
# [7] K-겹 교차 검증 (K-Fold Cross Validation)
# =============================================================================

# 데이터 다시 로드
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,0:60]
y = df.iloc[:,60]


# 교차 검증 설정 (5-Fold)
k=5
kfold = KFold(n_splits=k, shuffle=True)

# 각 폴드의 정확도를 저장할 리스트
acc_score = []


# 모델 생성 함수 정의
# 교차 검증 시 매번 새로운 모델을 생성해야 하므로 함수로 정의
def model_fn():
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# K-겹 교차 검증 수행
for train_index , test_index in kfold.split(X): # 데이터를 k개로 분할하여 반복
    # 학습 데이터와 검증 데이터 분리
    X_train , X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train , y_test = y.iloc[train_index], y.iloc[test_index]

    # 모델 생성 및 컴파일
    model = model_fn()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 모델 학습 (verbose=0으로 학습 과정 출력 생략)
    history=model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)

    # 정확도 평가 및 리스트에 저장
    accuracy = model.evaluate(X_test, y_test)[1] 
    acc_score.append(accuracy)

# 평균 정확도 계산
avg_acc_score = sum(acc_score)/k

# 최종 결과 출력
print('정확도:', acc_score)
print('정확도 평균:', avg_acc_score)