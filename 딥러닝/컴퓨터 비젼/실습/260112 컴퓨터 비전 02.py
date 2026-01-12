# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
import numpy as np               # 수치 연산 및 배열 처리를 위한 NumPy
import tensorflow as tf          # 딥러닝 프레임워크 TensorFlow
import matplotlib.pyplot as plt  # 데이터 시각화 및 그래프 출력을 위한 Matplotlib

# Keras 모델 및 레이어 관련 클래스 임포트
from tensorflow.keras.models import Sequential  # 순차적 모델 구성을 위한 클래스
from tensorflow.keras.layers import Dense       # 완전 연결 계층 (Fully Connected Layer)
from tensorflow.keras.optimizers import SGD     # 확률적 경사 하강법 (Stochastic Gradient Descent) 최적화 알고리즘
from tensorflow.keras.optimizers import Adam    # Adam (Adaptive Moment Estimation) 최적화 알고리즘

# Keras 내장 데이터셋 모듈 임포트
import tensorflow.keras.datasets as ds 

import os # 파일 경로 및 운영체제 상호작용을 위한 모듈

# =============================================================================
# [2] MNIST 데이터셋 로드 및 확인 (MNIST Data Loading & Inspection)
# =============================================================================
# MNIST 데이터셋 다운로드 및 로드
# x_train, y_train: 학습용 데이터 (이미지, 레이블)
# x_test, y_test: 테스트용 데이터 (이미지, 레이블)
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

# 데이터 형태(Shape) 출력
# 예: (60000, 28, 28) -> 60000개의 28x28 이미지
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# =============================================================================
# [3] MNIST 데이터 시각화 (MNIST Data Visualization)
# =============================================================================
plt.figure(figsize=(24, 3))         # 그래프 전체 크기 설정 (가로 24, 세로 3인치)
plt.suptitle('MNIST', fontsize=30)  # 전체 제목 설정

# 처음 10게의 이미지 출력
for i in range(10):
    plt.subplot(1, 10, i + 1)              # 1행 10열의 격자 중 i+1번째 위치 지정
    plt.imshow(x_train[i], cmap='gray')    # 이미지를 흑백(gray) 컬러맵으로 출력
    plt.xticks([]); plt.yticks([])         # 축 눈금 제거 (깔끔한 출력을 위해)
    plt.title(str(y_train[i]), fontsize=30) # 해당 이미지의 레이블(정답)을 제목으로 표시

# =============================================================================
# [4] CIFAR-10 데이터셋 로드 및 시각화 (CIFAR-10 Data Loading & Visualization)
# =============================================================================
# CIFAR-10 데이터셋 로드 (컬러 이미지)
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# CIFAR-10의 레이블(0~9)에 해당하는 클래스 이름 정의
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(24, 3))            # 그래프 크기 설정
plt.suptitle('CIFAR-10', fontsize=30)  # 전체 제목

# 처음 10개의 이미지 출력
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i])             # 컬러 이미지이므로 cmap 지정 없음
    plt.xticks([]); plt.yticks([])
    # y_train[i, 0]: 2차원 배열 형태이므로 첫 번째 인덱스 값 사용
    plt.title(class_names[y_train[i, 0]], fontsize=30)


# =============================================================================
# [5] 데이터 전처리 (Data Preprocessing) - MNIST for MLP
# =============================================================================
# MNIST 데이터 다시 로드 (초기화)
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

# 2차원 이미지(28x28)를 1차원 벡터(784)로 변환 (Flatten)
# MLP의 입력층에 넣기 위해 1줄로 펴는 작업
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 정규화 (Normalization)
# 픽셀 값(0~255)을 0.0 ~ 1.0 범위의 실수로 변환하여 학습 안정성 향상
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 원-핫 인코딩 (One-Hot Encoding)
# 정수형 레이블(0~9)을 10차원 벡터로 변환 (예: 2 -> [0,0,1,0,0,0,0,0,0,0])
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# =============================================================================
# [6] MLP 모델 구축 및 학습 (MLP Model Building & Training) - SGD
# =============================================================================
mlp = Sequential() # 순차적 모델 객체 생성

# 은닉층 (Hidden Layer) 추가
# units=512: 뉴런 개수 512개
# activation='tanh': 활성화 함수로 Hyperbolic Tangent 사용
# input_shape=(784,): 입력 데이터 형태 지정 (첫 번째 층 필수)
mlp.add(Dense(units=512, activation='tanh', input_shape=(784,)))

# 출력층 (Output Layer) 추가
# units=10: 클래스 개수(0~9)와 동일하게 10개 설정
# activation='softmax': 다중 클래스 분류를 위한 확률 값 출력
mlp.add(Dense(units=10, activation='softmax'))

# 모델 컴파일 (Compile)
# loss='MSE': 손실 함수로 평균 제곱 오차(Mean Squared Error) 사용
# optimizer=SGD: 최적화 알고리즘으로 SGD 사용 (학습률 0.01)
# metrics=['accuracy']: 학습 과정에서 정확도 모니터링
mlp.compile(loss='MSE', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# 모델 학습 (Fit)
# batch_size=128: 한 번에 학습할 데이터 양
# epochs=50: 전체 데이터를 50번 반복 학습
# validation_data: 학습 중 성능 검증을 위한 테스트 데이터 지정
# verbose=2: 학습 로그 출력 상세도 설정
mlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

# 모델 평가 (Evaluate)
# 테스트 데이터로 최종 성능 측정
res = mlp.evaluate(x_test, y_test, verbose=0)
print('정확률=', res[1] * 100) # 정확도 출력 (퍼센트)


# =============================================================================
# [7] 최적화 기법 비교 학습 (Optimizer Comparison) - Adam
# =============================================================================
# 데이터 다시 로드 및 전처리 (위와 동일한 과정 반복)
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 모델 재구축 (초기화)
mlp = Sequential()
mlp.add(Dense(units=512, activation='tanh', input_shape=(784,)))
mlp.add(Dense(units=10, activation='softmax'))

# Adam 옵티마이저 사용
# learning_rate=0.001: Adam의 기본 학습률
mlp.compile(loss='MSE', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 학습 수행
mlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

# 평가
res = mlp.evaluate(x_test, y_test, verbose=0)
print('정확률=', res[1] * 100)


# =============================================================================
# [8] SGD vs Adam 성능 비교 그래프 (Performance Comparison Graph)
# =============================================================================
# 비교를 위해 데이터 다시 로드 및 전처리
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ----------------- SGD 모델 -----------------
mlp_sgd = Sequential()
mlp_sgd.add(Dense(units=512, activation='tanh', input_shape=(784,)))
mlp_sgd.add(Dense(units=10, activation='softmax'))

mlp_sgd.compile(loss='MSE', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# 학습 결과(history)를 변수에 저장하여 그래프 그리기용으로 사용
hist_sgd = mlp_sgd.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

print('SGD 정확률=', mlp_sgd.evaluate(x_test, y_test, verbose=0)[1] * 100)

# ----------------- Adam 모델 -----------------
mlp_adam = Sequential()
mlp_adam.add(Dense(units=512, activation='tanh', input_shape=(784,)))
mlp_adam.add(Dense(units=10, activation='softmax'))

mlp_adam.compile(loss='MSE', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

hist_adam = mlp_adam.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

print('Adam 정확률=', mlp_adam.evaluate(x_test, y_test, verbose=0)[1] * 100)

# ----------------- 비교 그래프 출력 -----------------
# SGD 모델의 정확도 추이 (점선)
plt.plot(hist_sgd.history['accuracy'], 'r--')     # 학습 정확도 (Red Dashed)
plt.plot(hist_sgd.history['val_accuracy'], 'r')   # 검증 정확도 (Red Solid)

# Adam 모델의 정확도 추이 (파란선)
plt.plot(hist_adam.history['accuracy'], 'b--')    # 학습 정확도 (Blue Dashed)
plt.plot(hist_adam.history['val_accuracy'], 'b')  # 검증 정확도 (Blue Solid)

plt.title('Comparison of SGD and Adam optimizers') # 그래프 제목
plt.ylim((0.7, 1.0)) # Y축 범위 설정 (정확도 70% ~ 100% 구간 확대)
plt.xlabel('epochs') # X축 라벨
plt.ylabel('accuracy') # Y축 라벨
plt.legend(['train_sgd', 'val_sgd', 'train_adam', 'val_adam']) # 범례 표시
plt.grid() # 격자 표시
plt.show() # 그래프 출력


# =============================================================================
# [9] 심층 신경망 (Deep MLP) 구현 및 저장 (Deep MLP Implementation & Saving)
# =============================================================================
# 데이터 전처리
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 깊은 신경망 모델 구성
dmlp = Sequential()
# 은닉층 1: 1024개 뉴런, ReLU 활성화 함수
dmlp.add(Dense(units=1024, activation='relu', input_shape=(784,)))
# 은닉층 2: 512개 뉴런, ReLU
dmlp.add(Dense(units=512, activation='relu'))
# 은닉층 3: 512개 뉴런, ReLU
dmlp.add(Dense(units=512, activation='relu'))
# 출력층: 10개 클래스, Softmax
dmlp.add(Dense(units=10, activation='softmax'))

# 컴파일
# categorical_crossentropy: 다중 분류 문제에 적합한 손실 함수
# Adam 학습률 0.0001 (조금 더 세밀하게 학습)
dmlp.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# 학습 수행
hist = dmlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

# 평가
print('정확률=', dmlp.evaluate(x_test, y_test, verbose=0)[1] * 100)

# 모델 저장 (Saving Model)
# 스크립트 실행 디렉토리 경로 획득
script_dir = os.path.dirname(__file__)
# 모델을 'dmlp_trained.h5' 파일로 저장 (경로 결합 사용)
# 'data' 폴더가 없을 경우 에러가 날 수 있으므로 주의 필요 (여기서는 data 폴더 내 저장이 의도됨)
# 주의: data 폴더가 존재해야 함. '../data/...' 는 상대 경로 주의
dmlp.save(os.path.join(script_dir, 'data/dmlp_trained.h5'))


# =============================================================================
# [10] 학습 결과 그래프 시각화 (Visualization of Training Results)
# =============================================================================
# 정확도(Accuracy) 그래프
plt.plot(hist.history['accuracy'])      # 학습 정확도
plt.plot(hist.history['val_accuracy'])  # 검증 정확도
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.grid()
plt.show()

# 손실(Loss) 그래프
plt.plot(hist.history['loss'])      # 학습 손실
plt.plot(hist.history['val_loss'])  # 검증 손실
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'])
plt.grid()
plt.show()