# 필요한 라이브러리 임포트
import numpy as np # 수치 계산
import pandas as pd # 데이터 분석
import matplotlib.pyplot as plt # 시각화
import os # 파일 경로 조작

# 사이킷런 관련 라이브러리
from sklearn.model_selection import train_test_split # 데이터셋 분리
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score # 평가 지표
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve # 평가 지표 및 곡선
from sklearn.preprocessing import StandardScaler # 데이터 스케일링
from sklearn.preprocessing import Binarizer # 임계값 조절을 위한 이진화
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델

# 1. 데이터 로드 및 확인
# 데이터 파일 경로 설정 (현재 파일 위치 기준)
path = os.path.dirname(__file__)
load_file = os.path.join(path, 'diabetes.csv')

# CSV 파일 읽기
diabetes_data = pd.read_csv(load_file)

# 타겟 변수(Outcome)의 분포 확인 (0: 정상, 1: 당뇨병)
print(diabetes_data['Outcome'].value_counts())

# 데이터 앞부분 3개 행 확인
print(diabetes_data.head(3))

# 데이터 정보 확인 (결측치, 데이터 타입 등)
print(diabetes_data.info())


# 2. 평가 함수 정의
# 오차 행렬, 정확도, 정밀도, 재현율, F1 스코어, ROC AUC 값을 한 번에 출력
def get_clf_eval(y_test, pred=None, pred_proba=None):
  confusion = confusion_matrix(y_test, pred)
  accuracy = accuracy_score(y_test, pred)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)
  f1 = f1_score(y_test, pred)
  
  # ROC AUC는 예측 확률(positive label일 확률)을 사용
  roc_auc = roc_auc_score(y_test, pred_proba)
  
  print('오차 행렬')
  print(confusion)
  print(f'정확도: {accuracy:.4f} 정밀도: {precision:.4f} 재현율: {recall:.4f} F1: {f1:.4f} AUC: {roc_auc:.4f}')


# 3. 정밀도-재현율 곡선 시각화 함수 정의
def precision_recall_curve_plot(y_test=None, pred_proba_c1=None):
  # 임계값에 따른 정밀도, 재현율 추출
  precision, recall, thresholds = precision_recall_curve(y_test, pred_proba_c1)

  plt.figure(figsize=(8,6))
  thresholds_boundary = thresholds.shape[0]
  # 정밀도는 점선, 재현율은 실선으로 표시
  plt.plot(thresholds, precision[0:thresholds_boundary], linestyle='--', label='precision')
  plt.plot(thresholds, recall[0:thresholds_boundary], label='recall')
  
  # X축(임계값) 틱 설정
  start, end = plt.xlim()
  plt.xticks(np.round(np.arange(start, end, 0.1), 2))
  
  plt.xlabel('Threshold value')
  plt.ylabel('Precision and Recall value')
  plt.legend()
  plt.grid()
  plt.show()


# 4. 초기 모델 학습 및 평가 (데이터 전처리 전)
# 피처 데이터(마지막 컬럼 제외)와 타겟 데이터(마지막 컬럼) 분리
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

# 학습/테스트 데이터 분리 (stratify=y: 타겟 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

# 로지스틱 회귀 모델 생성 및 학습
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)

# 예측 수행 (결정값 및 확률값)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

# 평가 결과 출력
get_clf_eval(y_test, pred, pred_proba)

# 정밀도-재현율 곡선 시각화
pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
precision_recall_curve_plot(y_test, pred_proba_c1)

# 데이터 통계 확인 (min 값이 0인 이상치 확인용)
diabetes_data.describe()

# Glucose(포도당) 피처의 히스토그램 확인 -> 0인 값이 존재하는지 확인
plt.hist(diabetes_data['Glucose'], bins=10)
plt.show()


# 5. 데이터 전처리: 0값(결측치로 추정)을 평균값으로 대체
# 0값이 있으면 안되는 피처들 선정
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 전체 데이터 건수
total_count = diabetes_data['Glucose'].count()

# 각 피처별로 0인 데이터 건수와 비율 출력
for feature in zero_features:
  zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
  print(f'{feature} 0 건수는 {zero_count}, 퍼센트는 {100 * zero_count / total_count:.2f}')


# zero_features 리스트 내부에 있는 피처들의 0값을 해당 피처의 평균값으로 대체
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0, diabetes_data[zero_features].mean())


# 6. 스케일링 및 모델 재학습
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

# StandardScaler를 이용하여 피처 데이터 스케일링 (평균 0, 분산 1)
# 로지스틱 회귀의 경우 스케일링이 성능에 영향을 미침
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)


# 스케일링된 데이터로 다시 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=156, stratify=y)

# 모델 재학습
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)

# 예측 및 평가
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

get_clf_eval(y_test, pred, pred_proba)


# 7. 임계값(Threshold) 조정에 따른 성능 평가함수 정의
def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
  for custom_threshold in thresholds:
    # 주어진 임계값으로 Binarizer 생성 및 예측 수행
    binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
    custom_predict = binarizer.transform(pred_proba_c1)
    
    print(f'임곗값: {custom_threshold}')
    # 평가 수행
    get_clf_eval(y_test, custom_predict, pred_proba_c1)


# 여러 임계값 설정하여 성능 확인
thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds)


# 가장 적절하다고 판단된 임계값(예: 0.48)으로 최종 예측 및 평가
# 재현율을 높이면서 정확도와 정밀도를 적정 수준으로 유지하는 지점 선택
binarizer = Binarizer(threshold=0.48)

# 위에서 구한 임계값으로 최종 예측 수행
pred_th_048 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1, 1))

get_clf_eval(y_test, pred_th_048, pred_proba[:, 1])