# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 딥러닝 모델링 및 수치 계산에 필요한 핵심 라이브러리들을 불러옵니다.

# TensorFlow Keras: 딥러닝 모델을 쉽고 빠르게 구축하기 위한 고수준 API
# Sequential: 층(Layer)을 순차적으로 쌓아 모델을 만드는 방법
from tensorflow.keras.models import Sequential
# Dense: 모든 뉴런이 서로 연결된 완전 연결층(Fully Connected Layer)
from tensorflow.keras.layers import Dense, Input

# 수치 계산 및 시스템 관련 라이브러리
import numpy as np  # 배열, 행렬 연산 등 수치 계산
import os           # 파일 경로 제어 등 시스템 관련 기능


# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================

# 데이터 파일 경로 설정
# 현재 스크립트 파일의 경로를 기준으로 데이터 파일을 찾아 로드합니다.
# 이는 실행 환경에 상관없이 파일을 안정적으로 찾기 위함입니다.
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../data/ThoraricSurgery3.csv")

# CSV 파일 로드 (구분자: 콤마)
# np.loadtxt: 메모리 효율적인 텍스트 파일 로딩 함수
Data_set = np.loadtxt(file_path, delimiter=",")


# =============================================================================
# [3] 데이터 전처리 (Data Preprocessing)
# =============================================================================

# Feature(X)와 Target(y) 분리
# X: 환자의 진료 기록 데이터 (속성 1~16)
X = Data_set[:,0:16]
# y: 수술 1년 후 사망 여부 (속성 17, 0: 생존, 1: 사망) -> **Target Variable**
y = Data_set[:,16]


# =============================================================================
# [4] 모델 설계 및 층 설정 (Model Architecture)
# =============================================================================

# 순차적 모델(Sequential Model) 객체 생성
model = Sequential()

# [수정] 입력층 (Input Layer) 명시적 추가
# Keras 최신 버전에서는 첫 번째 층에 input_dim 대신 Input 객체를 사용하는 것을 권장합니다.
model.add(Input(shape=(16,)))

# 첫 번째 은닉층 (Hidden Layer 1)
# 60: 뉴런(노드)의 개수
# activation='relu': 활성화 함수로 ReLU 사용 (입력이 0보다 크면 그대로, 작으면 0)
model.add(Dense(60, activation='relu'))

# 출력층 (Output Layer)
# 1: 출력 뉴런의 개수 (이진 분류이므로 1개)
# activation='sigmoid': 활성화 함수로 시그모이드 사용 (출력을 0~1 사이의 확률값으로 변환)
model.add(Dense(1, activation='sigmoid'))


# =============================================================================
# [5] 모델 컴파일 및 학습 (Model Compilation & Training)
# =============================================================================

# 모델 컴파일 (학습 과정 설정)
# loss='binary_crossentropy': 이진 분류 문제에 적합한 오차 함수
# optimizer='adam': 효율적인 경사 하강법 알고리즘
# metrics=['accuracy']: 학습 중 모니터링할 성능 지표 (정확도)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습 (Training)
# X, y: 학습 데이터와 타겟
# epochs=60: 전체 데이터를 60번 반복 학습
# batch_size=16: 한 번의 학습에 사용할 데이터 샘플 수 (16개씩 끊어서 학습)
history=model.fit(X, y, epochs=60, batch_size=16)


# =============================================================================
# [6] 모델 평가 (Evaluation)
# =============================================================================

# 학습된 모델의 성능 평가 및 결과 출력
# model.evaluate(): 테스트 데이터(여기서는 학습 데이터 재사용)에 대한 손실값과 정확도 반환
# [1]: 정확도(Accuracy), [0]: 손실값(Loss)
print("\n Accuracy: %.4f" % (model.evaluate(X, y)[1]))
print("\n loss: %.4f" % (model.evaluate(X, y)[0]))