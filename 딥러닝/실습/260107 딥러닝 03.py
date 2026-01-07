# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석, 시각화 및 딥러닝 모델링에 필요한 핵심 라이브러리들을 불러옵니다.

import pandas as pd             # 데이터 처리 및 분석 (DataFrame)
import seaborn as sns           # 데이터 시각화 (통계적 그래프, pairplot 등)
import matplotlib.pyplot as plt # 데이터 시각화 (기본 그래프)
import os                       # 시스템 경로 및 파일 제어

# TensorFlow Keras: 딥러닝 모델 구축을 위한 고수준 API
from tensorflow.keras.models import Sequential # 순차적 모델 (Layer를 쌓는 방식)
from tensorflow.keras.layers import Dense      # 완전 연결층 (Fully Connected Layer)


# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================

# 현재 스크립트 파일의 경로를 기준으로 데이터 파일 경로 설정
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../data/iris3.csv")

# CSV 파일을 pandas DataFrame으로 로드
df = pd.read_csv(file_path)

# 데이터의 상위 5개 행 출력 (데이터 구조 확인)
print(df.head())


# =============================================================================
# [3] 데이터 시각화 (Data Visualization)
# =============================================================================

# 품종(species)에 따른 특성 간의 관계 시각화 (Pairplot)
# hue='species': 품종별로 색상을 다르게 표시하여 데이터 분포 확인
sns.pairplot(df, hue='species')
plt.show() # 그래프 출력


# =============================================================================
# [4] 데이터 전처리 (Data Preprocessing)
# =============================================================================

# Feature(X)와 Target(y) 분리
# X: 꽃받침 길이/너비, 꽃잎 길이/너비 (처음 4개 컬럼)
X = df.iloc[:,0:4]
# y: 붓꽃의 품종 (마지막 5번째 컬럼)
y = df.iloc[:,4]

# 분리된 데이터의 상위 5개 행 출력
print(X[0:5])
print(y[0:5])

# 원-핫 인코딩 (One-Hot Encoding)
# 문자열로 된 타겟(품종 이름)을 0과 1로 이루어진 벡터로 변환
# 예: "Iris-setosa" -> [1, 0, 0]
y = pd.get_dummies(y)
print(y[0:5])


# =============================================================================
# [5] 모델 설계 및 층 설정 (Model Architecture)
# =============================================================================

# 순차적 모델 객체 생성
model = Sequential()

# 첫 번째 은닉층 (Hidden Layer 1)
# input_dim=4: 입력 속성이 4개(꽃받침/꽃잎의 길이/너비)
# 12: 뉴런 개수, activation='relu'
model.add(Dense(12, input_dim=4, activation='relu'))

# 두 번째 은닉층 (Hidden Layer 2)
# 8: 뉴런 개수, activation='relu'
model.add(Dense(8, activation='relu'))

# 출력층 (Output Layer)
# 3: 붓꽃 품종이 3가지이므로 출력 뉴런은 3개
# activation='softmax': 다중 클래스 분류를 위한 확률 값 반환 (총합이 1)
model.add(Dense(3, activation='softmax'))

# 모델 구조 요약 출력
model.summary()


# =============================================================================
# [6] 모델 컴파일 및 학습 (Model Compilation & Training)
# =============================================================================

# 모델 컴파일
# loss='categorical_crossentropy': 다중 클래스 분류를 위한 오차 함수
# optimizer='adam': 최적화 알고리즘
# metrics=['accuracy']: 정확도 모니터링
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
# epochs=30: 전체 데이터셋을 30번 반복 학습
# batch_size=5: 5개의 샘플마다 가중치 갱신
history=model.fit(X, y, epochs=30, batch_size=5)


# =============================================================================
# [7] 모델 재학습 및 평가 (Model Retraining & Evaluation)
# =============================================================================
# (참고) 아래 코드는 위와 동일한 과정을 반복하여 모델을 새로 만들고 평가하는 예제입니다.

# 데이터 다시 로드 및 전처리
df = pd.read_csv(file_path)

X = df.iloc[:,0:4]
y = df.iloc[:,4]

# 타겟 원-핫 인코딩
y = pd.get_dummies(y)

# 새로운 모델 생성
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
print(model.summary())

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
history=model.fit(X, y, epochs=30, batch_size=5)

# 모델 평가 (테스트 데이터 없이 학습 데이터로 평가)
# score[0]: Loss, score[1]: Accuracy
score=model.evaluate(X, y)
print('Test accuracy:', score[1])
print('Test loss', score[0])