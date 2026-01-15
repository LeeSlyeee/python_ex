# =============================================================================
# [라이브러리 임포트]
# =============================================================================
import os                           # 운영체제 관련 기능을 제공하는 모듈 (파일 경로 조작 등)
import json                         # JSON 파일 입출력을 위한 모듈
import matplotlib.pyplot as plt     # 그래프 및 데이터 시각화 라이브러리
import tensorflow as tf             # 딥러닝 프레임워크 TensorFlow
from tensorflow.keras.optimizers import Adam # 최적화 알고리즘 Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 이미지 데이터 증강 및 제너레이터
from tensorflow.keras.models import Sequential, Model # 모델 구성을 위한 클래스 (순차적 API, 함수형 API)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input # 층(Layer) 클래스들
from tensorflow.keras.applications import MobileNetV2 # 사전 학습된 MobileNetV2 모델
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # MobileNetV2 전용 전처리 함수

# =============================================================================
# [1] 설정 및 데이터셋 준비 (Configuration & Dataset Preparation)
# =============================================================================

# 현재 실행 중인 스크립트 파일의 절대 경로를 구한 뒤 디렉토리 경로만 추출
script_dir = os.path.dirname(os.path.abspath(__file__))

# 분류할 클래스 정보가 담긴 JSON 파일의 전체 경로 생성
json_path = os.path.join(script_dir, 'garbage_classes.json')

# 학습할 이미지가 저장된 루트 디렉토리 경로 생성
dataset_dir = os.path.join(script_dir, 'Image_data')

# JSON 파일을 읽기 모드('r')로 열어서 파이썬 리스트로 로드
with open(json_path, 'r') as f:
    class_names = json.load(f)

# 로드된 타겟 클래스 목록 출력 (확인용)
print(f"Target Classes: {class_names}")

# -----------------------------------------------------------------------------
# ImageDataGenerator 설정
# -----------------------------------------------------------------------------
# MobileNetV2는 입력 픽셀 값을 [-1, 1] 범위로 스케일링하는 전처리를 사용함.
# 따라서 tensorflow.keras.applications.mobilenet_v2.preprocess_input 함수를 전처리 함수로 등록.
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # MobileNetV2에 맞는 전처리 함수 적용
    validation_split=0.2,   # 전체 데이터의 20%를 검증 데이터(Validation Set)로 사용
    rotation_range=20,      # 이미지를 랜덤하게 20도 내에서 회전 (데이터 증강)
    width_shift_range=0.2,  # 이미지를 좌우로 20% 내에서 이동 (데이터 증강)
    height_shift_range=0.2, # 이미지를 상하로 20% 내에서 이동 (데이터 증강)
    horizontal_flip=True    # 이미지를 수평(좌우) 반전 (데이터 증강)
)

# -----------------------------------------------------------------------------
# 학습용 데이터 생성기 (Train Generator)
# -----------------------------------------------------------------------------
# flow_from_directory 메서드를 사용해 디렉토리 구조로부터 데이터를 불러옴
train_generator = datagen.flow_from_directory(
    dataset_dir,            # 데이터가 위치한 루트 디렉토리
    target_size=(224, 224), # MobileNetV2의 기본 입력 크기인 224x224로 이미지 리사이징
    batch_size=32,          # 한 번에 학습할 이미지 수 (배치 크기)
    class_mode='categorical', # 다중 분류 문제이므로 원-핫 인코딩된 라벨 반환
    classes=class_names,    # JSON 파일 순서대로 클래스 레이블 매핑
    subset='training'       # 'validation_split' 설정 중 학습용(80%) 데이터 할당
)

# -----------------------------------------------------------------------------
# 검증용 데이터 생성기 (Validation Generator)
# -----------------------------------------------------------------------------
valid_generator = datagen.flow_from_directory(
    dataset_dir,            # 데이터가 위치한 루트 디렉토리
    target_size=(224, 224), # 입력 크기 동일하게 설정
    batch_size=32,          # 배치 크기 동일하게 설정
    class_mode='categorical', # 다중 분류 모드
    classes=class_names,    # 클래스 매핑
    subset='validation'     # 'validation_split' 설정 중 검증용(20%) 데이터 할당
)

# =============================================================================
# [2] 전이 학습 모델 설계 (Transfer Learning Model Design - MobileNetV2)
# =============================================================================

# 1. Base Model 로드 (기본 특징 추출기)
# include_top=False: 분류를 담당하는 최상위 완전 연결 층(Fully Connected Layer)을 제외하고 가져옴
# weights='imagenet': ImageNet 데이터셋으로 사전 학습된 가중치를 사용
# input_shape=(224, 224, 3): 모델의 입력 이미지 크기 (높이, 너비, 채널)
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

# 2. Base Model 동결 (Freezing)
# 사전 학습된 가중치가 학습 중에 업데이트(손상)되지 않도록 설정
# 즉, 특징 추출 능력은 그대로 유지하고 우리가 추가할 분류기 부분만 학습함
base_model.trainable = False

# 3. 새로운 분류기(Custom Head) 추가
inputs = Input(shape=(224, 224, 3)) # 입력 텐서 정의

# 입력 데이터를 Base Model에 통과시켜 특징 맵(Feature Map) 추출
# training=False: 배치 정규화(Batch Normalization) 층이 추론 모드로 동작하도록 설정
x = base_model(inputs, training=False)

# Global Average Pooling 층 추가
# 3차원 특징 맵(Height x Width x Channel)을 채널별 평균값을 구해 1차원 벡터로 변환
x = GlobalAveragePooling2D()(x)

# Dropout 층 추가
# 과적합(Overfitting)을 방지하기 위해 학습 시 20%의 뉴런을 무작위로 비활성화
x = Dropout(0.2)(x)

# 출력층 (Output Layer)
# len(class_names) 만큼의 노드를 생성 (여기서는 6개)
# 다중 분류를 위한 활성화 함수로 'softmax' 사용 (각 클래스별 확률 출력)
outputs = Dense(len(class_names), activation='softmax')(x)

# 최종 모델 생성
# 입력(inputs)부터 출력(outputs)까지 연결된 모델 정의
model = Model(inputs, outputs)

# 모델 구조 요약 출력 (레이어 구성 및 파라미터 수 확인)
model.summary()

# =============================================================================
# [3] 모델 컴파일 (Model Compilation)
# =============================================================================
# loss='categorical_crossentropy': 다중 클래스 분류의 손실 함수
# optimizer=Adam: 학습 최적화 알고리즘으로 Adam 사용 (learning_rate=0.001)
# metrics=['accuracy']: 학습 성능 평가 지표로 정확도 사용
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# =============================================================================
# [4] 모델 학습 (Model Training)
# =============================================================================
# 전이 학습은 이미 잘 학습된 특징 추출기를 사용하므로 적은 에폭(epoch)으로도 수렴이 빠름
epochs = 10 

# fit 메서드로 학습 시작
history = model.fit(
    train_generator, # 학습 데이터 제너레이터
    steps_per_epoch=train_generator.samples // train_generator.batch_size, # 한 에폭당 배치를 몇 번 돌릴지 계산
    epochs=epochs,   # 전체 데이터셋 반복 횟수
    validation_data=valid_generator, # 검증 데이터 제너레이터
    validation_steps=valid_generator.samples // valid_generator.batch_size, # 검증 시 배치를 몇 번 돌릴지 계산
    verbose=1        # 학습 진행 상황 출력 모드 (1: 로그바 표시)
)

# =============================================================================
# [5] 모델 저장 (Model Saving)
# =============================================================================
# 학습된 모델을 저장할 파일 경로 생성
model_save_path = os.path.join(script_dir, 'garbage_classification_model_transfer.h5')

# 모델 저장 (HDF5 포맷)
model.save(model_save_path)
print(f"Transfer Learning Model saved to {model_save_path}")

# =============================================================================
# [6] 결과 시각화 (Visualization of Results)
# =============================================================================
# 학습 과정에서 기록된 정확도와 손실 값을 가져옴
acc = history.history['accuracy']       # 학습 정확도
val_acc = history.history['val_accuracy'] # 검증 정확도
loss = history.history['loss']          # 학습 손실
val_loss = history.history['val_loss']  # 검증 손실

# x축 범위 생성 (0 ~ epochs-1)
epochs_range = range(len(acc))

# 그래프 크기 설정 (가로 12인치, 세로 4인치)
plt.figure(figsize=(12, 4))

# 1. 정확도 그래프 (왼쪽)
plt.subplot(1, 2, 1) # 1행 2열 중 첫 번째 영역
plt.plot(epochs_range, acc, label='Training Accuracy') # 학습 정확도 선 그래프
plt.plot(epochs_range, val_acc, label='Validation Accuracy') # 검증 정확도 선 그래프
plt.legend(loc='lower right') # 범례 위치 설정
plt.title('Training and Validation Accuracy (Transfer)') # 그래프 제목

# 2. 손실 그래프 (오른쪽)
plt.subplot(1, 2, 2) # 1행 2열 중 두 번째 영역
plt.plot(epochs_range, loss, label='Training Loss') # 학습 손실 선 그래프
plt.plot(epochs_range, val_loss, label='Validation Loss') # 검증 손실 선 그래프
plt.legend(loc='upper right') # 범례 위치 설정
plt.title('Training and Validation Loss (Transfer)') # 그래프 제목

# 그래프를 이미지 파일로 저장
plt.savefig(os.path.join(script_dir, 'training_history_transfer.png'))
print("History graph saved.")
# plt.show() # 로컬 실행 시 그래프를 화면에 출력하려면 주석 해제
