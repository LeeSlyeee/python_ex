# =============================================================================
# [라이브러리 임포트]
# =============================================================================
import os                           # 운영체제 관련 기능을 제공하는 모듈
import json                         # JSON 파일 입출력을 위한 모듈
import matplotlib.pyplot as plt     # 그래프 및 데이터 시각화 라이브러리
import tensorflow as tf             # 딥러닝 프레임워크 TensorFlow
from tensorflow.keras.optimizers import Adam # 최적화 알고리즘 Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 이미지 데이터 증강 및 제너레이터
from tensorflow.keras.models import Sequential, Model # 모델 구성을 위한 클래스
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input # 층(Layer) 클래스들
from tensorflow.keras.applications import MobileNetV2 # 사전 학습된 MobileNetV2 모델
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # MobileNetV2 전용 전처리 함수
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # 콜백 함수들

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
# ImageDataGenerator 설정 (데이터 증강 강화 - 개선판)
# -----------------------------------------------------------------------------
# [개선 1] 데이터 증강 강도 상향 조정
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # MobileNetV2 전처리
    validation_split=0.2,   # 검증 데이터 20%
    rotation_range=40,      # 회전 범위 확대 (30 -> 40)
    width_shift_range=0.3,  # 이동 범위 확대 (0.2 -> 0.3)
    height_shift_range=0.3, # 이동 범위 확대 (0.2 -> 0.3)
    shear_range=0.3,        # 기울기 범위 확대 (0.2 -> 0.3)
    zoom_range=0.3,         # 확대/축소 범위 확대 (0.2 -> 0.3)
    horizontal_flip=True,
    vertical_flip=False,    # 상하 반전은 쓰레기 데이터 특성상 혼란을 줄 수 있어 유지
    fill_mode='nearest'     # 회전/이동 시 빈 공간 채우는 방식
)

# -----------------------------------------------------------------------------
# 데이터 생성기 (Generators)
# -----------------------------------------------------------------------------
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=class_names,
    subset='training'
)

valid_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=class_names,
    subset='validation'
)

# =============================================================================
# [2] 전이 학습 모델 설계 (Transfer Learning Model Design - Improved)
# =============================================================================

# 1. Base Model 로드
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

# 2. 초기에는 Base Model 동결
base_model.trainable = False

# 3. 새로운 분류기(Custom Head) 추가
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False) # BatchNormalization 층은 추론 모드로 고정
x = GlobalAveragePooling2D()(x)
# [개선 2] Dropout 비율 상향 (0.2 -> 0.4) 과적합 방지 강화
x = Dropout(0.4)(x) 
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()

# =============================================================================
# [3] 콜백(Callback) 설정 - 학습 최적화 도구
# =============================================================================
# [개선 3] 모델 저장 경로 변경
model_save_path = os.path.join(script_dir, 'best_garbage_model_improved.h5')

callbacks = [
    # 검증 손실(val_loss)이 가장 낮은 최고의 모델만 저장
    ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', verbose=1),
    
    # 검증 손실이 10번의 에폭 동안 개선되지 않으면 학습 조기 종료
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', verbose=1),
    
    # 검증 손실이 3번의 에폭 동안 개선되지 않으면 학습률을 20%로 감소
    ReduceLROnPlateau(factor=0.2, patience=3, monitor='val_loss', verbose=1, min_lr=1e-6)
]

# =============================================================================
# [4] 1단계: Feature Extraction 학습 (Top Layer Training)
# =============================================================================
print("\n========== [Step 1] Training Output Layers (Frozen Base) ==========\n")

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001), # 초기 학습률
              metrics=['accuracy'])

initial_epochs = 10 # 1단계 에폭을 조금 더 늘려 충분히 학습 (5 -> 10)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=initial_epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# [5] 2단계: Fine-Tuning (미세 조정)
# =============================================================================
print("\n========== [Step 2] Fine-Tuning (Unfrozen Base) ==========\n")

# Base Model 동결 해제
base_model.trainable = True

# 모델의 전체 층 개수 확인
print(f"Number of layers in the base model: {len(base_model.layers)}")

# 상위 층 일부만 학습하고 하위 층(일반적인 특징)은 다시 동결
# MobileNetV2는 154개의 층이 있음. 상위 50개 정도만 풀어서 학습
fine_tune_at = 100 

# 최하위부터 fine_tune_at 층까지는 동결 유지
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 컴파일 다시 수행 (매우 낮은 학습률이 핵심!)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5), 
              metrics=['accuracy'])

# Fine-tuning 에폭 설정
fine_tune_epochs = 10 # 미세 조정 에폭도 충분히 (5 -> 10)
total_epochs = initial_epochs + fine_tune_epochs

# 이어서 학습 시 history 객체 연결을 위해 initial_epoch 설정
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # 1단계 학습이 끝난 시점부터 시작
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# [6] 결과 시각화 (Visualization)
# =============================================================================
# 1단계와 2단계 학습 기록 병합
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 6))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
# Fine-tuning 시작 지점 표시
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning', linestyle='--')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy (Improved)')

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, max(plt.ylim())])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning', linestyle='--')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (Improved)')

# 그래프 저장
graph_save_path = os.path.join(script_dir, 'training_history_improved.png')
plt.savefig(graph_save_path)
print(f"History graph saved to {graph_save_path}")
# plt.show()
