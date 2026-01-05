# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석, 시각화, 머신러닝 모델링에 필요한 핵심 라이브러리들을 불러옵니다.

# 수치 계산 및 데이터 조작을 위한 필수 라이브러리
import numpy as np  # 배열, 행렬 연산 등 수치 계산
import pandas as pd # 데이터프레임(표 형태) 처리를 위한 라이브러리

# 데이터 시각화를 위한 라이브러리
import matplotlib.pyplot as plt # 기본적인 그래프 그리기
import seaborn as sns           # matplotlib 기반의 통계적 시각화 도구

# 시스템 관련 라이브러리 (파일 경로 제어 등)
import os

# Scikit-learn(사이킷런) 관련 모듈 임포트
# 데이터 분할: 전체 데이터를 학습용(Train)과 테스트용(Test)으로 나누는 함수
from sklearn.model_selection import train_test_split
# 성능 평가 지표: 정확도, 정밀도, 재현율, ROC-AUC 등 분류 모델 평가 함수
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
# 데이터 전처리: 데이터의 스케일(범위)을 맞추는 도구 (평균 0, 분산 1로 변환)
from sklearn.preprocessing import StandardScaler
# 임계값(Threshold) 조절을 위한 이진화 도구
from sklearn.preprocessing import Binarizer
# 로지스틱 회귀 모델: 이진 분류(0 or 1)에 주로 사용되는 기본 모델
from sklearn.linear_model import LogisticRegression


# =============================================================================
# [2] 데이터 로드 및 확인 (Data Loading & Inspection)
# =============================================================================

# 현재 실행 중인 파일의 경로를 기반으로 CSV 파일 경로 생성
# (이렇게 하면 어디서 실행하든 경로 문제 없이 파일을 찾을 수 있습니다)
path = os.path.dirname(__file__)
load_file = os.path.join(path, 'heart_failure_clinical_records_dataset.csv')

# CSV 데이터를 pandas DataFrame으로 로드
df = pd.read_csv(load_file)

print("="*80)
print("1. 데이터 기본 정보")
print("="*80)
# 데이터의 상위 5개 행을 출력하여 구조 확인
print(df.head())

# <데이터 컬럼 설명>
# age: 환자의 나이
# anaemia: 환자의 빈혈증 여부 (0: 정상, 1: 빈혈)
# creatinine_phosphokinase: 크레아틴키나제 검사 결과
# diabetes: 당뇨병 여부 (0: 정상, 1: 당뇨)
# ejection_fraction: 박출계수 (%) - 심장이 피를 펌프질하는 효율
# high_blood_pressure: 고혈압 여부 (0: 정상, 1: 고혈압)
# platelets: 혈소판 수 (kiloplatelets/mL)
# serum_creatinine: 혈중 크레아틴 레벨 (mg/dL) - 신장 기능 지표
# serum_sodium: 혈중 나트륨 레벨 (mEq/L)
# sex: 성별 (0: 여성, 1: 남성)
# smoking: 흡연 여부 (0: 비흡연, 1: 흡연)
# time: 관찰 기간 (일)
# DEATH_EVENT: 사망 여부 (0: 생존, 1: 사망) -> **Target Variable**


# =============================================================================
# [3] 시각화 및 EDA (Exploratory Data Analysis)
# =============================================================================
# 데이터의 특성과 변수 간의 관계를 시각적으로 파악합니다.

# 한글 폰트 설정 (Mac 환경 기준: AppleGothic)
plt.rcParams['font.family'] = 'AppleGothic'
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# (1) 상관관계 히트맵 그리기
plt.figure(figsize=(12, 10))
# 그래프 가독성을 위해 영어 컬럼명을 한글로 매핑
kor_columns = {
    'age': '나이',
    'anaemia': '빈혈',
    'creatinine_phosphokinase': '크레아틴키나제',
    'diabetes': '당뇨',
    'ejection_fraction': '심장박출률',
    'high_blood_pressure': '고혈압',
    'platelets': '혈소판',
    'serum_creatinine': '혈중 크레아틴',
    'serum_sodium': '혈중 나트륨',
    'sex': '성별',
    'smoking': '흡연',
    'time': '기간',
    'DEATH_EVENT': '사망여부'
}
# df.corr(): 변수 간의 상관계수 계산 (-1 ~ 1)
# annot=True: 칸 안에 수치 표시, cmap='coolwarm': 색상 테마
sns.heatmap(df.rename(columns=kor_columns).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('상관관계 히트맵')
plt.savefig(os.path.join(path, 'correlation_heatmap.png'))
print("Correlation heatmap saved to correlation_heatmap.png")

# (2) 사망 여부에 따른 주요 변수 분포 (Histplot)
# 중요해 보이는 4가지 변수에 대해 생존자/사망자 분포 비교
df_kor = df.rename(columns=kor_columns)
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# 나이 분포
sns.histplot(x='나이', hue='사망여부', data=df_kor, kde=True, ax=ax[0][0])
ax[0][0].set_title('사망 여부에 따른 나이 분포')

# 심장박출률 분포
sns.histplot(x='심장박출률', hue='사망여부', data=df_kor, kde=True, ax=ax[0][1])
ax[0][1].set_title('사망 여부에 따른 심장박출률 분포')

# 혈중 크레아틴 분포
sns.histplot(x='혈중 크레아틴', hue='사망여부', data=df_kor, kde=True, ax=ax[1][0])
ax[1][0].set_title('사망 여부에 따른 혈중 크레아틴 분포')

# 관찰 기간 분포
sns.histplot(x='기간', hue='사망여부', data=df_kor, kde=True, ax=ax[1][1])
ax[1][1].set_title('사망 여부에 따른 관찰 기간 분포')

plt.tight_layout()
plt.savefig(os.path.join(path, 'distribution_plots.png'))
print("Distribution plots saved to distribution_plots.png")


# =============================================================================
# [4] 데이터 전처리 (Data Preprocessing)
# =============================================================================

# Feature(X)와 Target(y) 분리
# DEATH_EVENT 컬럼이 우리가 예측해야 할 정답입니다.
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 데이터 정규화 (Normalization/Standardization)
# 로지스틱 회귀는 데이터의 스케일(크기)에 민감하므로 정규화가 필수적입니다.
scaler = StandardScaler()
# 평균이 0, 표준편차가 1이 되도록 변환
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분리 (80% 학습, 20% 테스트)
# stratify=y: 타겟 값(0과 1)의 비율을 유지하며 분리 (데이터 불균형 문제 완화)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=11, stratify=y)


# =============================================================================
# [5] 모델 학습 (Model Training)
# =============================================================================

# 로지스틱 회귀 모델 객체 생성
lf_clf = LogisticRegression()
# 학습 수행 (fit)
lf_clf.fit(X_train, y_train)


# =============================================================================
# [6] 모델 평가 (Evaluation)
# =============================================================================

# 테스트 데이터에 대한 예측
pred = lf_clf.predict(X_test)                # 클래스 예측 (0 또는 1)
pred_proba = lf_clf.predict_proba(X_test)[:, 1] # 양성(사망, 1) 클래스일 확률 예측

# 여러 평가 지표를 한 번에 출력하는 사용자 정의 함수
def get_clf_eval(y_test, pred, pred_proba=None):
    confusion = confusion_matrix(y_test, pred) # 오차 행렬
    accuracy = accuracy_score(y_test, pred)    # 정확도
    precision = precision_score(y_test, pred)  # 정밀도
    recall = recall_score(y_test, pred)        # 재현율 (민감도)
    f1 = f1_score(y_test, pred)                # F1 점수 (정밀도와 재현율의 조화 평균)
    
    print('\n[오차 행렬]')
    print(confusion)
    print(f'\n정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}')
    
    # 확률값이 있으면 ROC AUC 점수도 출력
    if pred_proba is not None:
        roc_auc = roc_auc_score(y_test, pred_proba)
        print(f'AUC: {roc_auc:.4f}')

# 평가 결과 출력 (기본 임계값 0.5 사용)
get_clf_eval(y_test, pred, pred_proba)


# =============================================================================
# [7] 시각화: ROC Curve & PR Curve
# =============================================================================

# (1) ROC Curve 그리기
# 이진 분류 모델의 성능을 시각적으로 평가하는 곡선
fpr, tpr, thresholds_roc = roc_curve(y_test, pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC 곡선')
plt.plot([0, 1], [0, 1], 'k--', label='무작위 예측') # 기준선 (AUC=0.5)
plt.xlabel('FPR (1 - 특이도)')
plt.ylabel('TPR (재현율)')
plt.title('ROC 곡선')
plt.legend()
plt.savefig(os.path.join(path, 'roc_curve.png'))
print("ROC curve saved to roc_curve.png")

# (2) Precision-Recall Curve 그리기
# 데이터 불균형이 심할 때 더 유용한 평가 곡선
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, pred_proba)

plt.figure(figsize=(8, 6))
# 임계값에 따른 정밀도와 재현율 변화 시각화
plt.plot(thresholds_pr, precisions[:-1], linestyle='--', label='정밀도')
plt.plot(thresholds_pr, recalls[:-1], label='재현율')
plt.xlabel('임계값 (Threshold)')
plt.title('임계값에 따른 정밀도-재현율 곡선')
plt.legend()
plt.grid()
plt.savefig(os.path.join(path, 'precision_recall_curve.png'))
print("Precision-Recall curve saved to precision_recall_curve.png")


# =============================================================================
# [8] 임계값(Threshold) 튜닝 실험
# =============================================================================
# 분류 결정 임계값(기본 0.5)을 변경해가며 성능(정밀도 vs 재현율)이 어떻게 변하는지 실험합니다.

thresholds_list = [0.4, 0.45, 0.50, 0.55, 0.60]

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    print('\n[임계값 변화에 따른 성능 비교]')
    for custom_threshold in thresholds:
        # Binarizer를 사용하여 지정한 임계값보다 크면 1, 작으면 0으로 변환
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1.reshape(-1, 1))
        custom_predict = binarizer.transform(pred_proba_c1.reshape(-1, 1))
        
        print(f'\n임계값: {custom_threshold}')
        # 위에서 정의한 평가 함수 재사용
        get_clf_eval(y_test, custom_predict)

# 실험 실행
get_eval_by_threshold(y_test, pred_proba.reshape(-1, 1), thresholds_list)