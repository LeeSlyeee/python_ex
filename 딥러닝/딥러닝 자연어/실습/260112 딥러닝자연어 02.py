# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# Keras의 모델 및 레이어 모듈 임포트
from tensorflow.keras.models import Sequential                                  # 순차적 모델 클래스
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten # 다양한 신경망 층 (완전연결, 드롭아웃, 활성화, 임베딩, LSTM, 1D합성곱, 맥스풀링, 평탄화)
from tensorflow.keras.utils import to_categorical                               # 원-핫 인코딩 유틸리티
from tensorflow.keras.preprocessing import sequence                             # 시퀀스 데이터 처리 (패딩 등)
from tensorflow.keras.datasets import reuters                                   # 로이터 뉴스 데이터셋
from tensorflow.keras.datasets import imdb                                      # IMDB 영화 리뷰 데이터셋
from tensorflow.keras.callbacks import EarlyStopping                            # 학습 조기 종료 콜백
from tensorflow.keras.utils import plot_model                                   # 모델 구조 시각화 (선택 사항)

# Self-Attention 메커니즘 사용을 위한 외부 라이브러리
# pip install keras-self-attention 필요
from keras_self_attention import SeqSelfAttention

import numpy as np               # 수치 연산
import matplotlib.pyplot as plt  # 데이터 시각화



# =============================================================================
# [Part 1] 로이터 뉴스 카테고리 분류 - LSTM (Reuters News Classification)
# =============================================================================
print("\n" + "="*50)
print("[Part 1] 로이터 뉴스 카테고리 분류 (LSTM)")
print("="*50 + "\n")

# 1. 데이터 로드 (Data Loading)
# num_words=1000: 빈도수 상위 1,000개의 단어만 사용
# test_split=0.2: 전체 데이터의 20%를 테스트용으로 분리
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)


# 2. 데이터 탐색 (Data Exploration)
category = np.max(y_train) + 1       # 카테고리 개수 확인 (레이블이 0부터 시작하므로 +1)
print(category, '카테고리')
print(len(X_train), '학습용 뉴스 기사')
print(len(X_test), '테스트용 뉴스 기사')
print(X_train[0])                    # 첫 번째 기사 데이터(정수 시퀀스) 출력


# 3. 데이터 전처리 (Preprocessing)
# sequence.pad_sequences: 문장의 길이를 maxlen으로 통일 (부족하면 0으로 채움)
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test = sequence.pad_sequences(X_test, maxlen=100)

# to_categorical: 레이블을 원-핫 인코딩 (다중 분류 문제)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 4. 모델 설계 (Model Architecture)
model = Sequential()
# Embedding: 단어 인덱스를 밀집 벡터(Dense Vector)로 변환
# 1000: 입력 단어 집합의 크기 (num_words)
# 100: 임베딩 벡터의 차원
model.add(Embedding(1000, 100))

# LSTM (Long Short-Term Memory): 순환 신경망(RNN)의 일종으로 긴 시퀀스 학습에 유리
# 100: LSTM 셀의 히든 유닛(Hidden Unit) 개수
# activation='tanh': 활성화 함수로 tanh 사용
model.add(LSTM(100, activation='tanh'))

# 출력층 (Output Layer)
# 46: 뉴스 카테고리 개수 (클래스 수)
# activation='softmax': 다중 클래스 분류 확률 출력
model.add(Dense(46, activation='softmax'))


# 5. 모델 컴파일 (Model Compilation)
# loss='categorical_crossentropy': 다중 분류를 위한 손실 함수
# optimizer='adam': 아담 최적화 알고리즘
# metrics=['accuracy']: 정확도 모니터링
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 조기 종료 (Early Stopping) 설정
# monitor='val_loss': 검증 손실을 관찰
# patience=5: 검증 손실이 5회(Epoch) 동안 개선되지 않으면 학습 중단
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)


# 6. 모델 학습 (Training)
# batch_size=20: 한 번에 20개 샘플씩 학습
# epochs=200: 최대 200회 반복 (조기 종료 가능)
history = model.fit(X_train, y_train, batch_size=20, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping_callback])

# 7. 모델 평가 (Evaluation)
print(f"\n Test Accuracy: {model.evaluate(X_test, y_test)[1]:.4f}")


# 8. 학습 결과 시각화 (Visualization)
y_vloss = history.history['val_loss'] # 검증 손실
y_loss = history.history['loss']      # 학습 손실

x_len = np.arange(len(y_loss))        # 에포크 수
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.title('Reuters Classification Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



# =============================================================================
# [Part 2] IMDB 영화 리뷰 감성 분류 - CNN + LSTM (Hybrid Model)
# =============================================================================
print("\n" + "="*50)
print("[Part 2] IMDB 영화 리뷰 감성 분류 (CNN + LSTM)")
print("="*50 + "\n")

# 1. 데이터 로드 (Data Loading)
# num_words=5000: 빈도수 상위 5,000개 단어 사용
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# 2. 전처리 (Preprocessing)
# maxlen=500: 리뷰 길이를 500단어로 맞춤
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# 3. 모델 설계 (Model Architecture)
model = Sequential()
# 임베딩 층
model.add(Embedding(5000, 100))

# 드롭아웃 (Dropout)
# 0.5(50%)의 뉴런을 무작위로 비활성화하여 과적합 방지
model.add(Dropout(0.5))

# Conv1D (1D Convolution): 텍스트의 지역적 특징 추출 (n-gram 효과)
# 64: 필터 개수
# 5: 커널 크기 (5개 단어를 윈도우로 봄)
# padding='valid': 패딩 없이 유효한 영역만 합성곱 수행
# strides=1: 1칸씩 이동
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))

# MaxPooling1D: 중요 특징 추출 및 차원 축소
# pool_size=4: 4개 중 가장 큰 값 선택
model.add(MaxPooling1D(pool_size=4))

# LSTM 층: 시계열(순서) 정보 학습
# 55: 히든 유닛 개수
model.add(LSTM(55))

# 출력층 (Output Layer)
# 1: 이진 분류 (긍정/부정)
# activation='sigmoid': 0~1 사이 확률 출력
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 입력 형태 명시 (모델 요약 출력을 위해)
model.build(input_shape=(None, 500)) # (배치크기, 시퀀스길이) -> 여기서는 maxlen을 500이 아닌 100으로 호출한 흔적 수정 (코드는 maxlen=500으로 전처리됨)
# 주의: 위 코드에서 pad_sequences를 maxlen=500으로 했으므로 input_shape=(None, 500)이 맞습니다.
# 다만 원본 코드 흔적(100)을 수정하여 500으로 표기하는 것이 정확하나,
# Embedding 층은 입력 길이를 자동으로 처리하므로 (None, None)으로도 동작합니다.
# 여기서는 데이터 전처리에 맞춰 500으로 간주하고 진행.
model.summary()


# 4. 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 조기 종료 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

# 6. 학습
history = model.fit(X_train, y_train, batch_size=40, epochs=100, validation_split=0.25, callbacks=[early_stopping_callback])

# 7. 평가
print(f"\n Test Accuracy: {model.evaluate(X_test, y_test)[1]:.4f}")


# 8. 시각화
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.title('IMDB CNN+LSTM Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



# =============================================================================
# [Part 3] IMDB 영화 리뷰 감성 분류 - Self-Attention (With LSTM)
# =============================================================================
print("\n" + "="*50)
print("[Part 3] IMDB 영화 리뷰 감성 분류 (LSTM + Self-Attention)")
print("="*50 + "\n")

# 1. 데이터 로드 (Data Loading)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# 2. 전처리 (Preprocessing)
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# 3. 모델 설계 (Model Architecture)
model = Sequential()
# 임베딩 층: 차원을 500으로 설정 (더 풍부한 표현)
model.add(Embedding(5000, 500))

# 드롭아웃
model.add(Dropout(0.5))

# LSTM 층
# return_sequences=True: 모든 시점(Time Step)의 은닉 상태를 출력해야 Attention 메커니즘이 각 시점을 참고할 수 있음
model.add(LSTM(64, return_sequences=True))

# Self-Attention 층
# 문맥을 고려하여 중요한 단어에 가중치를 부여하는 메커니즘
# attention_activation='sigmoid': 어텐션 가중치 계산에 사용할 활성화 함수
model.add(SeqSelfAttention(attention_activation="sigmoid"))

# 드롭아웃 추가
model.add(Dropout(0.5))

# 평탄화 (Flatten): 3차원 출력 -> 1차원
model.add(Flatten())

# 출력층
model.add(Dense(1))
model.add(Activation('sigmoid'))


# 4. 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 조기 종료 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

# 6. 학습
history = model.fit(X_train, y_train, batch_size=40, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping_callback])

# 7. 평가
print(f"\n Test Accuracy: {model.evaluate(X_test, y_test)[1]:.4f}")

# 8. 시각화
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.title('IMDB Self-Attention Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
