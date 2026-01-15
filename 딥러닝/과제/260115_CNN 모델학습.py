# =============================================================================
# [라이브러리 임포트]
# =============================================================================
import os                           # 운영체제 명령(파일 경로 등) 사용을 위한 모듈
import json                         # JSON 데이터 파싱을 위한 모듈
import numpy as np                  # 수치 연산 및 배열 처리를 위한 라이브러리
import matplotlib.pyplot as plt     # 그래프 및 데이터 시각화 도구
import tensorflow as tf             # 구글의 딥러닝 프레임워크 텐서플로우
from tensorflow.keras.optimizers import Adam # 가중치 업데이트를 위한 최적화 알고리즘 (Adam)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 이미지 데이터 전처리 및 증강 도구
from tensorflow.keras.models import Sequential # 레이어를 순차적으로 쌓는 모델 생성 방식
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense # CNN 모델 구성에 필요한 레이어들

# =============================================================================
# [1] 설정 및 데이터셋 준비 (Settings & Dataset Preparation)
# =============================================================================

# 현재 스크립트 파일이 위치한 디렉토리의 절대 경로 확보
script_dir = os.path.dirname(os.path.abspath(__file__))

# 클래스 이름이 저장된 JSON 파일 경로 지정
json_path = os.path.join(script_dir, 'garbage_classes.json')

# 학습에 사용할 이미지 데이터가 있는 루트 폴더 경로 지정
dataset_dir = os.path.join(script_dir, 'Image_data')

# JSON 파일에서 클래스 목록을 읽어옴 (예: ["glass", "paper", ...])
with open(json_path, 'r') as f:
    class_names = json.load(f)

# 읽어온 클래스 이름 출력
print(f"Target Classes: {class_names}")

# -----------------------------------------------------------------------------
# ImageDataGenerator 설정
# -----------------------------------------------------------------------------
# 딥러닝 모델 학습을 위해 이미지 픽셀 값을 0~255에서 0~1 사이로 정규화(Rescaling)
# validation_split=0.2 옵션을 주어 전체 데이터 중 20%를 검증용으로 자동 분리
datagen = ImageDataGenerator(
    rescale=1.0/255,      # 1/255를 곱해 정규화
    validation_split=0.2  # 전체 데이터의 20%를 검증용으로 사용
)

# -----------------------------------------------------------------------------
# 학습용 데이터 생성기 (Train Generator)
# -----------------------------------------------------------------------------
# 지정된 디렉토리(dataset_dir)에서 이미지를 불러와 학습 데이터 배치 생성
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(300, 300),   # 모델 입력 크기에 맞춰 이미지 리사이징 (300x300)
    batch_size=32,            # 배치 크기: 한 번에 학습할 이미지 수
    class_mode='categorical', # 다중 분류 문제: 정답 라벨을 원-핫 인코딩 벡터로 반환
    classes=class_names,      # JSON 파일 기준 클래스 이름 및 순서 지정
    subset='training'         # 앞서 설정한 validation_split에 따라 80%를 학습용으로 사용
)

# -----------------------------------------------------------------------------
# 검증용 데이터 생성기 (Validation Generator)
# -----------------------------------------------------------------------------
# 지정된 디렉토리(dataset_dir)에서 검증 데이터 배치 생성
valid_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(300, 300),   # 학습 데이터와 동일한 크기로 리사이징
    batch_size=32,            # 배치 크기
    class_mode='categorical', # 다중 분류 모드
    classes=class_names,      # 클래스 매핑 정보
    subset='validation'       # 나머지 20%를 검증용으로 사용
)

# 생성된 클래스 인덱스 맵핑 확인 (예: {'cardboard': 0, 'glass': 1, ...})
print("Class Indices:", train_generator.class_indices)


# =============================================================================
# [2] CNN 모델 설계 (CNN Model Design)
# =============================================================================
# Sequential API를 사용하여 레이어들을 순서대로 쌓아 올림
model = Sequential([
    # 1. 첫 번째 합성곱 층 (Convolutional Layer)
    # 16개의 필터, 3x3 커널 크기, ReLU 활성화 함수 사용
    # input_shape=(300, 300, 3): 입력 이미지 크기 지정 (가로 300, 세로 300, 채널 3-RGB)
    Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    # 맥스 풀링 층: 2x2 사이즈로 특징 맵의 크기를 절반으로 줄여 계산량 감소 및 특징 요약
    MaxPool2D(2, 2),

    # 2. 두 번째 합성곱 층
    # 필터 개수를 32개로 늘려 더 다양한 특징 추출
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D(2, 2),

    # 3. 세 번째 합성곱 층
    # 필터 개수 64개로 증가
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),

    # 4. 네 번째 합성곱 층
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),

    # 5. 다섯 번째 합성곱 층
    # 깊은 네트워크를 구성하여 추상적인 이미지 특징 학습
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),

    # 6. 평탄화 층 (Flatten)
    # 2D/3D 형태의 특징 맵을 1D 벡터(일렬)로 펼쳐서 완전 연결 층(Dense)에 전달
    Flatten(),

    # 7. 은닉층 (Hidden Layer)
    # 512개의 뉴런(노드)을 가진 완전 연결 층, 활성화 함수는 ReLU
    Dense(512, activation='relu'),

    # 8. 출력층 (Output Layer)
    # 분류할 클래스 개수(len(class_names)=6)만큼의 노드 생성
    # 다중 분류이므로 각 클래스에 속할 확률을 출력하는 'softmax' 활성화 함수 사용
    Dense(len(class_names), activation='softmax')
])

# 모델의 전체 구조와 파라미터 개수를 요약해서 출력
model.summary()


# =============================================================================
# [3] 모델 컴파일 (Model Compilation)
# =============================================================================
# 모델이 학습할 방식 지정
# loss='categorical_crossentropy': 다중 클래스 분류 문제의 오차 계산 함수
# optimizer=Adam(learning_rate=0.001): 오차를 줄이는 방향으로 가중치를 업데이트하는 알고리즘
# metrics=['accuracy']: 학습 중간 점검 시 '정확도'를 기준으로 성능 평가
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


# =============================================================================
# [4] 모델 학습 (Model Training)
# =============================================================================
# 전체 데이터셋을 몇 번 반복 학습할지(epochs) 설정
epochs = 15

# model.fit() 함수로 실제 학습 시작
history = model.fit(
    train_generator, # 학습용 데이터 제너레이터 전달
    steps_per_epoch=train_generator.samples // train_generator.batch_size, # 한 epoch당 배치를 몇 번 가져올지 계산
    epochs=epochs,   # 지정된 횟수만큼 반복 학습
    validation_data=valid_generator, # 매 epoch 종료 후 성능 평가를 위한 검증 데이터
    validation_steps=valid_generator.samples // valid_generator.batch_size, # 검증용 배치 횟수
    verbose=1        # 학습 진행 상황을 막대 그래프(progress bar)로 표시
)


# =============================================================================
# [5] 모델 저장 (Model Saving)
# =============================================================================
# 학습이 완료된 모델을 파일(.h5)로 저장할 경로 생성
model_save_path = os.path.join(script_dir, 'garbage_classification_model.h5')

# 모델 저장 (가중치 및 모델 구조 포함)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


# =============================================================================
# [6] 학습 결과 시각화 (Visualization of Results)
# =============================================================================
# history 객체에서 에폭별 정확도와 손실 값을 추출
acc = history.history['accuracy']       # 학습 데이터 정확도
val_acc = history.history['val_accuracy'] # 검증 데이터 정확도
loss = history.history['loss']          # 학습 데이터 손실값
val_loss = history.history['val_loss']  # 검증 데이터 손실값

# 에폭 범위 생성 (0 ~ 14)
epochs_range = range(len(acc))

# 그래프 그리기 준비 (가로 12, 세로 4 크기)
plt.figure(figsize=(12, 4))

# 왼쪽: 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right') # 범례 위치
plt.title('Training and Validation Accuracy')

# 오른쪽: 손실(Loss) 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right') # 범례 위치
plt.title('Training and Validation Loss')

# 그래프를 이미지 파일로 저장
plt.savefig(os.path.join(script_dir, 'training_history.png'))
print(f"Training history graph saved to {os.path.join(script_dir, 'training_history.png')}")
# plt.show() # 화면 출력을 원할 경우 주석 해제

# =============================================================================
# [7] 모델 평가 (Model Evaluation)
# =============================================================================
# 검증 데이터셋을 사용하여 최종 성능 평가
results = model.evaluate(valid_generator)
print("test loss, test acc:", results)
