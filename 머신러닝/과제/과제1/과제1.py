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
load_file = os.path.join(path, 'heart_failure_clinical_records_dataset.csv')

# 데이터프레임으로 로드
df = pd.read_csv(load_file)

# 데이터프레임의 처음 5개 행 확인
print(df.head())

# <데이터 컬럼 설명>
# age: 환자의 나이
# anaemia: 환자의 빈혈증 여부 (0: 정상, 1: 빈혈)
# creatinine_phosphokinase: 크레아틴키나제 검사 결과
# diabetes: 당뇨병 여부 (0: 정상, 1: 당뇨)
# ejection_fraction: 박출계수 (%)
# high_blood_pressure: 고혈압 여부 (0: 정상, 1: 고혈압)
# platelets: 혈소판 수 (kiloplatelets/mL)
# serum_creatinine: 혈중 크레아틴 레벨 (mg/dL)
# serum_sodium: 혈중 나트륨 레벨 (mEq/L)
# sex: 성별 (0: 여성, 1: 남성)
# smoking: 흡연 여부 (0: 비흡연, 1: 흡연)
# time: 관찰 기간 (일)
# DEATH_EVENT: 사망 여부 (0: 생존, 1: 사망)
import seaborn as sns

# 한글 폰트 설정 (Mac 환경 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 1.5 탐색적 데이터 분석 (EDA) - 시각화를 통한 인사이트 도출

# (1) 상관관계 히트맵
plt.figure(figsize=(12, 10))
# 한글 필드명 매핑
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
sns.heatmap(df.rename(columns=kor_columns).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('상관관계 히트맵')
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap saved to correlation_heatmap.png")

# (2) 사망 여부에 따른 주요 변수 분포 (나이, 심장박출률, 혈중 크레아틴, 기간)
df_kor = df.rename(columns=kor_columns)
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(x='나이', hue='사망여부', data=df_kor, kde=True, ax=ax[0][0])
ax[0][0].set_title('사망 여부에 따른 나이 분포')

sns.histplot(x='심장박출률', hue='사망여부', data=df_kor, kde=True, ax=ax[0][1])
ax[0][1].set_title('사망 여부에 따른 심장박출률 분포')

sns.histplot(x='혈중 크레아틴', hue='사망여부', data=df_kor, kde=True, ax=ax[1][0])
ax[1][0].set_title('사망 여부에 따른 혈중 크레아틴 분포')

sns.histplot(x='기간', hue='사망여부', data=df_kor, kde=True, ax=ax[1][1])
ax[1][1].set_title('사망 여부에 따른 관찰 기간 분포')

plt.tight_layout()
plt.savefig('distribution_plots.png')
print("Distribution plots saved to distribution_plots.png")

# 2. 데이터 전처리
# Feature와 Target 분리 (DEATH_EVENT가 타겟)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 정규화 (StandardScaler) - 거리 기반 알고리즘이나 로지스틱 회귀에서 중요
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분리 (80% 학습, 20% 테스트)
# stratify=y 옵션을 사용하여 타겟 비율을 유지하며 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=11, stratify=y)

# 3. 모델 학습 (로지스틱 회귀)
lf_clf = LogisticRegression() # Solver default is lbfgs
lf_clf.fit(X_train, y_train)

# 4. 예측 및 평가
pred = lf_clf.predict(X_test)
pred_proba = lf_clf.predict_proba(X_test)[:, 1] # 양성(1) 클래스에 대한 확률

# 평가 함수 정의
def get_clf_eval(y_test, pred, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    print('\n[오차 행렬]')
    print(confusion)
    print(f'\n정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}')
    
    if pred_proba is not None:
        roc_auc = roc_auc_score(y_test, pred_proba)
        print(f'AUC: {roc_auc:.4f}')

# 평가 결과 출력
get_clf_eval(y_test, pred, pred_proba)

# 5. 시각화 (ROC Curve)
fpr, tpr, thresholds_roc = roc_curve(y_test, pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC 곡선')
plt.plot([0, 1], [0, 1], 'k--', label='무작위 예측') # 대각선
plt.xlabel('FPR (1 - 특이도)')
plt.ylabel('TPR (재현율)')
plt.title('ROC 곡선')
plt.legend()
plt.savefig('roc_curve.png')
print("ROC curve saved to roc_curve.png")

# (추가) Precision-Recall Curve
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(thresholds_pr, precisions[:-1], linestyle='--', label='정밀도')
plt.plot(thresholds_pr, recalls[:-1], label='재현율')
plt.xlabel('임계값')
plt.title('임계값에 따른 정밀도-재현율 곡선')
plt.legend()
plt.grid()
plt.savefig('precision_recall_curve.png')
print("Precision-Recall curve saved to precision_recall_curve.png")

# (추가) 임계값(Threshold) 변화에 따른 성능 평가 (Binarizer 활용)
thresholds_list = [0.4, 0.45, 0.50, 0.55, 0.60]

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    print('\n[임계값 변화에 따른 성능 비교]')
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1.reshape(-1, 1))
        custom_predict = binarizer.transform(pred_proba_c1.reshape(-1, 1))
        
        print(f'\n임계값: {custom_threshold}')
        get_clf_eval(y_test, custom_predict)

get_eval_by_threshold(y_test, pred_proba.reshape(-1, 1), thresholds_list)