# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
import os                           # 파일 경로 및 시스템 제어
import pandas as pd                 # 데이터 프레임 처리 (본 코드에서는 미사용)
import cv2                          # OpenCV 라이브러리 (이미지 읽기 및 처리)
import numpy as np                  # 수치 연산 및 배열 처리
import matplotlib.pyplot as plt     # 그래프 및 이미지 시각화
import matplotlib.image as mpimg    # 이미지 파일 읽기 및 출력
import tensorflow as tf             # 딥러닝 프레임워크 TensorFlow
from tensorflow.keras.optimizers import Adam # 최적화 알고리즘 (Adam)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 이미지 데이터 증강 및 전처리

# =============================================================================
# [2] 데이터 경로 설정 및 탐색 (Data Path Setting & Exploration)
# =============================================================================
# 현재 스크립트 파일의 디렉토리 경로 확보
script_dir = os.path.dirname(__file__)

# 말(Horse)과 사람(Human) 이미지 데이터셋 경로 설정
# 학습 데이터는 'dataset/horse-or-human/train' 폴더 아래에 위치
train_horse_dir = os.path.join(script_dir, 'dataset/horse-or-human/train/horses')
train_human_dir = os.path.join(script_dir, 'dataset/horse-or-human/train/humans')

# 각 디렉토리 내의 파일 이름 리스트 추출
train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

# 파일 전체 경로 리스트 생성 (시각화 용도)
horse_files = [train_horse_dir + '/' + f for f in train_horse_names]
human_files = [train_human_dir + '/' + f for f in train_human_names]

# 데이터 개수 확인 출력
print('horse:', len(train_horse_names)) # 말 이미지 개수 출력
print('human:', len(train_human_names)) # 사람 이미지 개수 출력


# =============================================================================
# [3] 데이터 시각화 (Data Visualization)
# =============================================================================
# 2행 5열의 서브플롯 생성 (총 10개 이미지 출력)
fig, axes = plt.subplots(2,5, figsize=(12,4))
ax = np.reshape(axes, -1) # 반복문을 위해 2차원 배열을 1차원으로 평탄화

# 말 이미지 5개 + 사람 이미지 5개를 순서대로 출력
for i, path in enumerate(horse_files[:5] + human_files[:5]):
    img = plt.imread(path) # 이미지 파일 읽기
    ax[i].axis('off')      # 축 정보(눈금 등) 숨기기
    ax[i].imshow(img)      # 이미지 표시

plt.show() # 그래프 출력


# =============================================================================
# [4] CNN 모델 설계 (CNN Model Design)
# =============================================================================
# Sequential API를 사용하여 레이어를 순차적으로 적재
model = tf.keras.models.Sequential([
    # 1. 첫 번째 합성곱 층 (Convolutional Layer)
    # 16개의 필터, 3x3 커널 사용, 입력 이미지 크기는 (300, 300, 3) 컬러 이미지
    # 활성화 함수로 ReLU 사용
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    # 맥스 풀링 (Max Pooling): 2x2 영역에서 최대값 추출 -> 이미지 크기 절반으로 축소
    tf.keras.layers.MaxPool2D(2, 2),

    # 2. 두 번째 합성곱 층
    # 필터 수를 32개로 늘려 더 복잡한 특징 추출
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    # 3. 세 번째 합성곱 층
    # 필터 수를 64개로 증가
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    # 4. 네 번째 합성곱 층
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    # 5. 다섯 번째 합성곱 층
    # 깊은 네트워크로 추상적인 특징 학습
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    # 6. 평탄화 (Flatten)
    # 3차원 특징 맵을 1차원 벡터로 변환하여 완전 연결 층에 전달
    tf.keras.layers.Flatten(),

    # 7. 은닉층 (Hidden Layer)
    # 512개의 뉴런을 가진 완전 연결 층 (Dense Layer)
    tf.keras.layers.Dense(512, activation='relu'),

    # 8. 출력층 (Output Layer)
    # 이진 분류(Binary Classification)이므로 1개의 노드 사용
    # 활성화 함수로 Sigmoid 사용 (0~1 사이의 확률 값 출력)
    # 0에 가까우면 클래스 A(예: 말), 1에 가까우면 클래스 B(예: 사람)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 구조 요약 출력 (파라미터 수 등 확인)
model.summary()


# =============================================================================
# [5] 모델 컴파일 (Model Compilation)
# =============================================================================
# loss='binary_crossentropy': 이진 분류 문제의 표준 손실 함수
# optimizer=Adam(learning_rate=0.001): 아담 최적화 알고리즘 사용, 학습률 0.001 설정
# metrics=['accuracy']: 평가 지표로 정확도 사용
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


# =============================================================================
# [6] 데이터 전처리 및 제너레이터 생성 (Data Preprocessing & Generator)
# =============================================================================
# ImageDataGenerator를 사용하여 픽셀 값을 0~1 사이로 정규화 (Rescaling)
train_datagen = ImageDataGenerator(rescale=1/255)

# 학습 및 검증 데이터 디렉토리 경로 설정
train_dir = os.path.join(script_dir, 'dataset/horse-or-human/train')
valid_dir = os.path.join(script_dir, 'dataset/horse-or-human/validation')

# flow_from_directory: 디렉토리에서 이미지를 불러와 배치를 생성
# target_size=(300, 300): 모델 입력 크기에 맞춰 이미지 리사이징
# batch_size=128: 한 번에 학습할 이미지 수
# class_mode='binary': 이진 분류 문제 설정
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), batch_size=128, class_mode='binary')

valid_generator = train_datagen.flow_from_directory(valid_dir, target_size=(300, 300), batch_size=128, class_mode='binary')


# =============================================================================
# [7] 모델 학습 (Model Training)
# =============================================================================
# steps_per_epoch=8: 전체 학습 데이터 약 1027장을 배치크기 128로 나누면 약 8번의 스텝 필요
# epochs=15: 전체 데이터를 15번 반복 학습
# verbose=1: 학습 진행 상황 출력
history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1) 


# =============================================================================
# [8] 학습 결과 시각화 (Visualization of Results)
# =============================================================================
# 정확도(accuracy)와 손실(loss) 변화 추이를 그래프로 출력
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model accuracy & loss')
plt.xlabel('Epoch')
plt.ylabel('value')
plt.legend(['accuracy', 'loss'], loc='center right')
plt.show()


# =============================================================================
# [9] 모델 평가 (Model Evaluation)
# =============================================================================
# 검증 데이터셋을 사용하여 모델 성능 최종 평가
results = model.evaluate(valid_generator)
print("test loss, test acc:", results)


# =============================================================================
# [10] 사용자 입력 이미지 예측 (Prediction w/ Custom Images)
# =============================================================================
# 테스트용 이미지가 있는 디렉토리 설정
dlist = os.path.join(script_dir, 'dataset/horse-or-human/test')

# 테스트 디렉토리 내의 파일 목록 가져오기
predict_list = os.listdir(dlist)
print(predict_list)

# OpenCV를 사용하여 이미지 읽기 및 전처리
# os.path.join을 사용하여 안전하게 경로 결합
img = [cv2.imread(os.path.join(dlist, i)) for i in predict_list]
# 모델 입력 크기인 (300, 300)으로 리사이징. 주의: OpenCV는 (너비, 높이) 순서
img = [cv2.resize(i, (300,300)) for i in img]
# 리스트를 넘파이 배열로 변환
img = np.array(img)
# 픽셀 값을 0~1 사이로 정규화 및 실수형 변환
img = img.astype('float32') / 255.0

# 예측 수행
predictions = model.predict(img)

# 임계값(cutoff) 설정: 0.5 이상이면 클래스 1, 미만이면 클래스 0
cutoff = .5
# 예측 결과 출력 (True/False)
print(predictions >= cutoff)