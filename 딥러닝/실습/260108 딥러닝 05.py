# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석, 파일 제어, 시각화 및 딥러닝 모델링에 필요한 핵심 라이브러리들을 불러옵니다.

import warnings
warnings.filterwarnings('ignore') # 실행 시 불필요한 경고 메시지(FutureWarning 등)를 숨깁니다.

# TensorFlow Keras: 딥러닝 모델 구축 및 최적화 도구
from tensorflow.keras.models import Sequential # 순차적 모델 생성 클래스
from tensorflow.keras.layers import Dense      # 완전 연결층 (Fully Connected Layer)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # 모델 최적화를 위한 콜백(저장, 조기종료)
from tensorflow.keras.callbacks import LambdaCallback # 사용자 정의 콜백 함수 생성을 위한 도구

# Scikit-learn: 데이터 전처리
from sklearn.model_selection import train_test_split # 학습/테스트 데이터셋 분리 함수

# 데이터 처리 및 시각화 도구
import os                       # 시스템 경로 및 파일 제어
import pandas as pd             # 데이터 처리 (DataFrame)
import numpy as np              # 수치 연산 (배열 처리)
import matplotlib.pyplot as plt # 그래프 시각화
import tensorflow as tf         # TensorFlow 로깅 등 활용


# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================

# 현재 스크립트 파일의 경로를 기준으로 데이터 파일 경로 설정
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../data/wine.csv")

# 헤더가 없는 CSV 파일 로드 (첫 번째 행도 데이터로 인식)
df = pd.read_csv(file_path, header=None)

# 데이터프레임 전체 출력 확인 (데이터 구조 파악)
print(df)

# Feature(X)와 Target(y) 분리
# X: 0번부터 11번 컬럼까지 (12개의 속성 데이터)
X = df.iloc[:,0:12]
# y: 12번 컬럼 (클래스: 1=Red Wine, 0=White Wine 등)
y = df.iloc[:,12]

# 학습 데이터(80%)와 테스트 데이터(20%)로 분리
# test_size=0.2: 전체의 20%를 테스트 셋으로 할당
# shuffle=True: 데이터를 무작위로 섞어서 분할
# stratify=y: 클래스 분포를 유지하여 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)


# =============================================================================
# [3] 기본 모델 설계 및 학습 (Basic Model Training)
# =============================================================================
# 모델 체크포인트나 조기 종료 없이 기본적으로 학습을 수행하는 과정입니다.

# 모델 구조 설계
model = Sequential()
# 은닉층 1: 30개 노드, 입력 속성 12개, 활성화 함수 ReLU
model.add(Dense(30, input_dim=12, activation='relu'))
# 은닉층 2: 12개 노드, 활성화 함수 ReLU
model.add(Dense(12, activation='relu'))
# 은닉층 3: 8개 노드, 활성화 함수 ReLU
model.add(Dense(8, activation='relu'))
# 출력층: 1개 노드, 이진 분류이므로 활성화 함수 Sigmoid 사용
model.add(Dense(1, activation='sigmoid'))

# 모델 구조 요약 출력
model.summary()

# 모델 컴파일
# loss='binary_crossentropy': 이진 분류 오차 함수
# optimizer='adam': 최적화 알고리즘
# metrics=['accuracy']: 정확도 모니터링
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
# validation_split=0.25: 학습 데이터(X_train) 중 25%를 떼어내어 검증(Validation) 데이터로 사용
# epochs=50: 50번 반복 학습
# batch_size=500: 한 번에 500개의 샘플씩 학습
history=model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25)

# 테스트 데이터로 최종 성능 평가
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])


# =============================================================================
# [4] 모델 업데이트 및 저장 (Model Checkpoint)
# =============================================================================
# 학습 도중 성능을 모니터링하고, 모델을 파일로 저장하는 방법을 실습합니다.

# 데이터 다시 로드 (실습 독립성을 위해 초기화)
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,0:12]
y = df.iloc[:,12]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# 모델 재설계 (위와 동일 구조)
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 저장 경로 설정
# {epoch:02d}: 에포크 번호를 2자리 숫자로 표시 (예: 01, 02)
# {val_accuracy:.4f}: 검증 정확도를 소수점 4자리까지 표시
modelpath = os.path.join(script_dir, '../data/model/all/{epoch:02d}-{val_accuracy:.4f}.keras')

# ModelCheckpoint 콜백 설정
# filepath: 모델이 저장될 경로 패턴
# verbose=1: 모델이 저장될 때마다 터미널에 로그 메시지 출력
checkpointer = ModelCheckpoint(filepath=modelpath, verbose=1)

# 학습 실행
# callbacks=[checkpointer]: 학습 도중 매 에포크 끝마다 checkpointer를 실행하여 모델 저장
history=model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25, verbose=0, callbacks=[checkpointer])


# 평가 수행
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])


# =============================================================================
# [5] 학습 과정 시각화 (Visualization)
# =============================================================================
# 학습이 진행됨에 따라 학습셋 오차와 검증셋 오차가 어떻게 변하는지 그래프로 확인합니다.

# 커스텀 로그 함수 정의
# 50 에포크마다 학습 상태를 출력하는 함수입니다.
def custom_log(epoch, logs):
    if (epoch + 1) % 50 == 0:
        num_batches = len(X_train) // 500
        print(f"Epoch {epoch+1}/2000")
        tf.print(f"{num_batches}/{num_batches} ━━━━━━━━━━━━━━━━━━━━ "
        f"accuracy: {logs['accuracy']:.4f} - loss: {logs['loss']:.4f} - "
        f"val_accuracy: {logs['val_accuracy']:.4f} - val_loss: {logs['val_loss']:.4f}")

# LambdaCallback을 사용하여 커스텀 함수를 콜백으로 변환
show_status = LambdaCallback(on_epoch_end=custom_log)

# 긴 학습 실행 (2000 에포크)
# verbose=0: 기본 로그를 끄고, 커스텀 콜백으로만 로그 출력 확인
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=0, callbacks=show_status)


# 학습 결과(history)를 데이터프레임으로 변환하여 확인
hist_df=pd.DataFrame(history.history)
print(hist_df)

# 오차값(Loss) 추출
y_vloss=hist_df['val_loss'] # 검증셋(Validation) 오차: 빨간색 점
y_loss=hist_df['loss']      # 학습셋(Train) 오차: 파란색 점

# 그래프 그리기
x_len = np.arange(len(y_loss)) # X축: 에포크 수
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Val_loss')
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Train_loss')

plt.legend(loc='upper right') # 범례 위치 설정
plt.xlabel('epoch')           # X축 라벨
plt.ylabel('loss')            # Y축 라벨
plt.show()                    # 그래프 출력


# =============================================================================
# [6] 조기 종료 (Early Stopping) & 최적 모델 저장
# =============================================================================
# 과적합을 방지하기 위해 학습을 자동으로 중단하고, 최적의 모델만 저장하는 실전 기법입니다.

# 데이터 다시 로드
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,0:12]
y = df.iloc[:,12]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# 모델 재설계
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 조기 종료(EarlyStopping) 콜백 설정
# monitor='val_loss': 검증셋의 오차를 감시 대상으로 설정
# patience=20: 검증 오차가 줄어들지 않더라도 20번의 에포크까지는 기다려줌 (그 이후에도 개선 없으면 중단)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)


# 최적 모델 저장 경로 설정
modelpath = os.path.join(script_dir, '../data/model/bestmodel.keras')

# 체크포인트(ModelCheckpoint) 콜백 설정
# monitor='val_loss': 검증 오차를 기준으로 판단
# save_best_only=True: 이전 저장된 모델보다 성능이 좋아졌을 때만 덮어쓰기 (최적 모델 유지)
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 학습 실행
# callbacks 리스트에 두 개의 콜백(조기종료, 체크포인트)을 모두 전달
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=1,
callbacks=[early_stopping_callback,checkpointer])


# 최종 테스트 데이터 평가
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])