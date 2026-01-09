# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# TensorFlow Keras: 딥러닝 모델 구축을 위한 핵심 모듈
from tensorflow.keras.models import Sequential # 레이어를 순차적으로 쌓는 모델 클래스
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # 각종 레이어 (완전연결, 드롭아웃, 평탄화, 합성곱, 풀링)
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping # 학습 제어 콜백 (모델 저장, 조기 종료)
from tensorflow.keras.datasets import mnist # MNIST 손글씨 데이터셋
from tensorflow.keras.utils import to_categorical # 원-핫 인코딩 유틸리티

# 데이터 처리 및 시각화 라이브러리
import matplotlib.pyplot as plt # 그래프 및 이미지 시각화
import numpy as np              # 수치 연산 및 배열 처리
import os                       # 파일 경로 및 시스템 제어
import sys                      # 시스템 출력 제어

# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================
# MNIST 데이터셋 불러오기
# X_train, y_train: 학습용 데이터와 정답(레이블)
# X_test, y_test: 테스트용 데이터와 정답(레이블)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# =============================================================================
# [3] 데이터 탐색 및 전처리 (Data Exploration & Preprocessing)
# =============================================================================
# 데이터셋의 크기(형상) 확인
print(f"학습셋 이미지 수 : {X_train.shape[0]} 개") # 학습용 이미지 60,000개
print(f"테스트셋 이미지 수 : {X_test.shape[0]} 개") # 테스트용 이미지 10,000개

# 첫 번째 학습용 이미지 시각화 (확인용)
# cmap='Greys': 흑백 이미지로 출력
plt.imshow(X_train[0], cmap='Greys')
plt.show()

# 이미지 픽셀 값 출력 (0~255 사이의 정수값 확인)
# 28x28 픽셀의 값을 텍스트 형태로 시각화하여 데이터 구조 이해
for x in X_train[0]:
    for i in x:
        sys.stdout.write("%-3s" % i) # 각 픽셀 값을 3자리 폭으로 출력
    sys.stdout.write('\n')


# 학습 데이터 차원 변환 (Reshape) 및 정규화 (Normalization)
# 2차원 이미지(28x28)를 1차원 배열(784)로 펼침 (Dense 레이어 입력용)
# .astype('float64'): 정수형 데이터를 실수형으로 변환
# / 255: 픽셀 값(0~255)을 0~1 사이의 값으로 정규화하여 학습 효율 증대
X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255

# 테스트 데이터도 동일하게 전처리 수행
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

# 타겟 레이블(정답) 클래스 확인
print(f"class : {y_train[0]}") # 첫 번째 데이터의 정답 레이블 출력

# 원-핫 인코딩 (One-Hot Encoding)
# 정수형 클래스(0~9)를 10차원 벡터로 변환 (예: 5 -> [0,0,0,0,0,1,0,0,0,0])
# 다중 분류 문제(Multi-class Classification)에서 주로 사용
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(y_train[0]) # 변환된 레이블 확인




# =============================================================================
# [4] MLP 모델 설계 및 학습 (MLP Model Design & Training)
# =============================================================================

# 데이터 다시 로드 (MLP 모델 학습용 초기화)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 전처리 수행
# MLP 입력에 맞게 1차원(784)으로 변환 및 0~1 사이 값으로 정규화
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

# 정답 레이블 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



# 모델 구조 설정
# 모델 구조 설계 (Sequential API 사용)
model = Sequential()
# 은닉층 (Hidden Layer): 512개의 노드
# input_dim=784: 입력 데이터가 784개의 값을 가짐 (28x28 픽셀)
# activation='relu': 활성화 함수로 ReLU 사용
model.add(Dense(512, input_dim=784, activation='relu'))

# 출력층 (Output Layer): 10개의 노드 (0~9 숫자 분류)
# activation='softmax': 다중 분류 문제이므로 각 클래스에 속할 확률을 출력
model.add(Dense(10, activation='softmax'))

# 모델 구조 요약 출력
model.summary()



# 모델 컴파일
# 모델 컴파일 (Compile)
# loss='categorical_crossentropy': 다중 분류 문제의 손실 함수
# optimizer='adam': 효율적인 경사 하강법 최적화 알고리즘
# metrics=['accuracy']: 학습 중 모델 성능을 정확도로 평가
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 저장 및 조기 종료 설정
# 모델 저장 경로 설정
# script_dir: 현재 실행 중인 파이썬 파일의 위치
script_dir = os.path.dirname(__file__)
# ../data/model: 상위 디렉터리(..)로 이동 후 data/model 폴더 접근 (상대 경로)
modelpath = os.path.join(script_dir, "../data/model/MNIST_MLP.keras")

# ModelCheckpoint: validation loss가 개선될 때마다 모델 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
verbose=1, save_best_only=True)
# EarlyStopping: validation loss가 10회(patience=10) 이상 개선되지 않으면 학습 중단
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델 학습 (Fit)
# validation_split=0.25: 학습 데이터의 25%를 검증용으로 사용
# batch_size=200: 한 번에 학습할 데이터 샘플 수
# epochs=30: 전체 데이터를 30회 반복 학습
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])

# 정확도 출력
# 테스트 정확도 출력
# model.evaluate(): 테스트 데이터로 모델 성능 평가 [loss, accuracy] 반환
print(f"\n Test Accuracy: {(model.evaluate(X_test, y_test)[1]):.4f}")


# 학습 결과 그래프로 시각화
# 검증셋(Validation Set) 오차
y_vloss = history.history['val_loss']
# 학습셋(Training Set) 오차
y_loss = history.history['loss']

x_len = np.arange(len(y_loss)) # x축 좌표 (에포크 수)
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss') # 검증 오차 (빨간색)
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss') # 학습 오차 (파란색)


plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



# =============================================================================
# [5] CNN 모델 설계 및 학습 (CNN Model Design & Training)
# =============================================================================

# 데이터 다시 로드 (CNN 모델 학습용 초기화)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# CNN 입력을 위한 전처리 (4차원 변환)
# (샘플 수, 가로, 세로, 채널 수) -> (N, 28, 28, 1)
# 흑백 이미지이므로 채널 수는 1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# 정답 레이블 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# CNN 모델 구조 설정
# CNN(Convolutional Neural Network) 모델 구조 설계
model = Sequential()

# 첫 번째 합성곱 층 (Convolutional Layer)
# Filters=32: 32개의 특징 맵(Feature Map) 생성
# kernel_size=(3, 3): 3x3 크기의 필터(커널) 사용
# input_shape=(28, 28, 1): 입력 이미지 크기 명시 (가로 28, 세로 28, 채널 1)
# activation='relu': 활성화 함수로 ReLU 사용
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))

# 두 번째 합성곱 층
# Filters=64: 64개의 특징 맵 생성
model.add(Conv2D(64, (3, 3), activation='relu'))

# 맥스 풀링 층 (Max Pooling Layer)
# pool_size=(2,2): 2x2 영역에서 가장 큰 값만 추출하여 이미지 크기 축소 (중요 정보 유지, 연산량 감소)
model.add(MaxPooling2D(pool_size=(2,2)))

# 드롭아웃 (Dropout)
# 0.25(25%)의 뉴런을 무작위로 비활성화하여 과적합(Overfitting) 방지
model.add(Dropout(0.25))

# 평탄화 (Flatten)
# 2차원(3차원) 특징 맵을 1차원 배열로 펼침 (Dense 층에 입력하기 위함)
model.add(Flatten())

# 완전 연결 층 (Fully Connected Layer)
# 128개의 노드 사용
model.add(Dense(128, activation='relu'))

# 드롭아웃 추가 (50% 비활성화)
model.add(Dropout(0.5))

# 출력층
# 10개의 노드 (0~9 분류), Softmax 함수 사용
model.add(Dense(10, activation='softmax'))



# 모델 컴파일
# 모델 컴파일
# 평가 지표로 'accuracy' 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 저장 및 조기 종료 설정
# CNN 모델 저장 경로 설정 (상대 경로 사용)
modelpath = os.path.join(script_dir, "../data/model/MNIST_CNN.keras")
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델 학습
# 학습 및 검증 데이터 분리 (25% 검증용)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])

# 정확도 출력
# 최종 테스트 정확도 출력
print(f"\n Test Accuracy: {(model.evaluate(X_test, y_test)[1]):.4f}")


# 학습 오차와 검증 오차 비교 그래프 시각화
y_vloss = history.history['val_loss'] # 검증셋 오차
y_loss = history.history['loss']      # 학습셋 오차

x_len = np.arange(len(y_loss)) # 에포크 수
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()