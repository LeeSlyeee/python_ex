# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석 및 시각화, 딥러닝 모델링에 필요한 라이브러리들을 불러옵니다.

import pandas as pd             # 데이터 처리 및 분석 (DataFrame)
import matplotlib.pyplot as plt # 데이터 시각화 (기본 그래프)
import seaborn as sns           # 데이터 시각화 (통계적 그래프)
import os                       # 시스템 경로 및 파일 제어

# 경고 메시지 무시 설정 (버전 호환성 경고 등 불필요한 출력 방지)
import warnings
warnings.filterwarnings("ignore")

# TensorFlow Keras 관련 모듈 임포트
from tensorflow.keras.models import Sequential # 순차적 모델 생성
from tensorflow.keras.layers import Dense      # 완전 연결층(Fully Connected Layer)

# 데이터를 학습용과 테스트용으로 나누기 위한 함수 불러오기
from sklearn.model_selection import train_test_split


# =============================================================================
# [2] 데이터 로드 (Data Loading)
# =============================================================================

# 현재 스크립트 파일의 경로를 기준으로 데이터 파일 경로 설정
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "../data/pima-indians-diabetes3.csv")

# CSV 파일을 pandas DataFrame으로 로드
df = pd.read_csv(file_path)

# 데이터의 상위 5개 행 출력 (데이터 구조 확인)
print(df.head(5))

# 'diabetes' 컬럼의 값 분포 확인 (0: 정상, 1: 당뇨병)
print(df["diabetes"].value_counts())

# 데이터의 통계적 요약 정보 출력 (평균, 표준편차, 최소/최대값 등)
print(df.describe())

# 변수 간의 상관계수 출력
print(df.corr())


# =============================================================================
# [3] 데이터 시각화 (Data Visualization)
# =============================================================================

# (1) 상관관계 히트맵 (Heatmap)
colormap = plt.cm.gist_heat   # 그래프 색상 테마 설정
plt.figure(figsize=(12,12))   # 그래프 크기 설정

# 상관계수를 히트맵으로 시각화 (vmax: 색상의 최대값, annot: 수치 표시)
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show() # 그래프 출력

# (2) 주요 변수별 당뇨병 발병 분포 (Histogram)
# 혈장 포도당 농도(plasma)에 따른 정상/당뇨 분포
plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]], bins=30,
histtype='barstacked', label=['normal','diabetes'])
plt.legend() # 범례 표시
# plt.show() # (주석: 필요시 주석 해제하여 출력)

# 체질량 지수(BMI)에 따른 정상/당뇨 분포
plt.hist(x=[df.bmi[df.diabetes==0], df.bmi[df.diabetes==1]], bins=30, histtype='barstacked', label=['normal','diabetes'])
plt.legend()
# plt.show()


# =============================================================================
# [4] 데이터 전처리 (Data Preprocessing)
# =============================================================================

# 데이터 다시 로드 (분석 단계와 독립적으로 처리하기 위함)
df = pd.read_csv(file_path)

# Feature(X)와 Target(y) 분리
# iloc: 위치 기반 인덱싱 (모든 행, 0~7번 컬럼)
X = df.iloc[:,0:8]
# 타겟 변수 (모든 행, 8번 컬럼 -> 당뇨병 여부)
y = df.iloc[:,8]


# =============================================================================
# [4.5] 데이터 분리 (Train-Test Split) - 새로 추가된 단계
# =============================================================================

# 전체 데이터를 8:2 비율로 분리
# 학습은 X_train, y_train으로 하고 / 평가는 X_test, y_test로 하게 됩니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # 20%를 테스트용으로 떼어둠
    shuffle=True,    # 데이터를 무작위로 섞음
    random_state=42  # 결과를 고정하기 위한 시드값
)

# =============================================================================
# [5] 모델 설계 및 층 설정 (Model Architecture) - 동일
# =============================================================================
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid', name='Dense_3'))

# =============================================================================
# [6] 모델 컴파일 및 학습 (Model Compilation & Training) - 변수 변경
# =============================================================================
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ★ 변경점: X, y 대신 -> X_train, y_train 사용 (공부용 데이터)
history = model.fit(X_train, y_train, epochs=100, batch_size=5)

# =============================================================================
# [7] 모델 평가 (Evaluation) - 변수 변경
# =============================================================================

# ★ 변경점: X, y 대신 -> X_test, y_test 사용 (시험용 데이터)
# 모델이 한 번도 본 적 없는 데이터로 평가를 해야 진짜 실력이 나옵니다.
print("\n=== 테스트 데이터 평가 결과 ===")
score = model.evaluate(X_test, y_test)
print(f"Loss: {score[0]:.4f}")
print(f"Accuracy: {score[1]:.4f}")