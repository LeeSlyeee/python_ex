# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석, 시각화, 머신러닝 모델링에 필요한 핵심 라이브러리들을 불러옵니다.

# 수치 계산 및 데이터 조작을 위한 필수 라이브러리
import numpy as np  # 배열, 행렬 연산 등 수치 계산
import pandas as pd # 데이터프레임(표 형태) 처리를 위한 라이브러리

# 데이터 시각화를 위한 라이브러리
import matplotlib.pyplot as plt # 기본적인 그래프 그리기
import seaborn as sns           # matplotlib 기반의 통계적 시각화 도구 (더 예쁜 그래프)

# 시스템 관련 라이브러리 (파일 경로 제어 등)
import os

# Scikit-learn(사이킷런) 관련 모듈 임포트
# 데이터 분할: 전체 데이터를 학습용(Train)과 테스트용(Test)으로 나누는 함수
from sklearn.model_selection import train_test_split
# 데이터 전처리: 범주형 변수(문자열)를 숫자형으로 변환하는 인코더
from sklearn.preprocessing import LabelEncoder
# 성능 평가 지표: 정확도, 정밀도, 재현율, F1 점수 등 분류 모델 평가 함수
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 추가 평가 도구: 혼동 행렬, 분류 리포트, ROC-AUC 점수, ROC 곡선
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# XGBoost 라이브러리 (Gradient Boosting 알고리즘의 고성능 구현체)
from xgboost import XGBClassifier

# 경고 메시지 제어 (실행 시 불필요한 경고 무시)
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정: 그래프에서 한글 폰트 깨짐 방지 (Mac: AppleGothic, Win: Malgun Gothic)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# =============================================================================
# [2] 데이터 로드 및 기본 확인 (Data Loading & Inspection)
# =============================================================================

# 현재 파일의 실행 경로를 기준으로 CSV 파일 경로 생성
path = os.path.dirname(__file__)
data_file = os.path.join(path, 'xAPI-Edu-Data.csv')

# CSV 데이터를 pandas DataFrame으로 로드
df = pd.read_csv(data_file)

print("="*80)
print("1. 데이터 기본 정보")
print("="*80)
# 데이터의 크기(행, 열) 확인
print(f"데이터 크기: {df.shape}")
# 데이터프레임에 포함된 모든 컬럼명 출력
print(f"\n데이터 컬럼:\n{df.columns.tolist()}")
# 데이터의 상위 5개 행을 출력하여 실제 데이터 구조 확인
print(f"\n첫 5개 행:\n{df.head()}")
# 데이터프레임의 요약 정보(데이터 타입, Non-Null Count 등) 출력
print(f"\n데이터 정보:\n{df.info()}")
# 수치형 변수들의 기술 통계량(평균, 표준편차, 4분위수 등) 확인
print(f"\n기술 통계:\n{df.describe()}")
# 각 컬럼별 결측치(Null) 개수 확인 -> 결측치가 있다면 전처리 필요
print(f"\n결측치:\n{df.isnull().sum()}")


# =============================================================================
# [3] 타겟 변수 탐색 (Target Variable EDA)
# =============================================================================
# 예측해야 할 목표 변수('Class')의 분포를 확인합니다.
# 클래스 불균형이 있는지 파악하는 중요한 단계입니다.

print("\n" + "="*80)
print("2. 타겟 변수 (Class) 분포")
print("="*80)

# Class 컬럼의 고유값별 개수 세기 (L, M, H)
print(df['Class'].value_counts())

# 타겟 변수 분포 시각화 (막대 그래프)
plt.figure(figsize=(8, 6))
# 막대 그래프 색상을 지정하여 시각적으로 구분
df['Class'].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('학생 성적 등급 분포') # 그래프 제목
plt.xlabel('성적 등급')          # X축 레이블
plt.ylabel('학생 수')            # Y축 레이블
plt.xticks(rotation=0)           # X축 눈금 글자 회전 (수평 유지)
plt.tight_layout()               # 여백 자동 조정
# 그래프 이미지 파일로 저장
plt.savefig(os.path.join(path, 'class_distribution.png'))
print("타겟 분포 그래프 저장: class_distribution.png")


# =============================================================================
# [4] 수치형 변수 탐색 (Numeric Features EDA)
# =============================================================================
# 성적 등급(Class)에 따라 학생들의 행동 패턴(수치형 변수)이 어떻게 다른지 분석합니다.

print("\n" + "="*80)
print("3. 주요 수치형 변수 분석")
print("="*80)

# 분석할 주요 수치형 컬럼 리스트 정의
numeric_cols = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
# 시각화를 위한 한글 매핑 딕셔너리
feature_kor_names = {
    'raisedhands': '손을 들 횟수', 
    'VisITedResources': '과목 공지 확인 횟수',
    'AnnouncementsView': '공지사항 확인 횟수', 
    'Discussion': '토론 참여 횟수',
    'gender': '성별', 
    'NationalITy': '국적', 
    'PlaceofBirth': '태어난 국가',
    'StageID': '학교 단계', 
    'GradeID': '성적 등급(ID)', 
    'SectionID': '반 이름',
    'Topic': '과목', 
    'Semester': '학기', 
    'Relation': '보호자 관계',
    'ParentAnsweringSurvey': '부모 설문 참여', 
    'ParentschoolSatisfaction': '부모 만족도',
    'StudentAbsenceDays': '결석 횟수'
}

# 2x2 그리드의 서브플롯(그래프 영역) 생성
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# 2차원 배열 형태인 axes를 1차원 배열로 평탄화 (반복문처리를 쉽게 하기 위해)
axes = axes.flatten()

# 각 수치형 변수에 대해 반복하며 상자 그림(Boxplot) 그리기
for i, col in enumerate(numeric_cols):
    # 성적 등급(Class)별로 수치 변수의 분포를 상자 그림으로 시각화
    sns.boxplot(data=df, x='Class', y=col, ax=axes[i], palette='Set2')
    axes[i].set_title(f'성적 등급별 {feature_kor_names[col]} 분포')  # 각 서브플롯 제목 설정
    axes[i].set_xlabel('성적 등급')       # x축 레이블 설정
    axes[i].set_ylabel(feature_kor_names[col]) # y축 레이블 설정

plt.tight_layout()
# 분석 결과 그래프 저장
plt.savefig(os.path.join(path, 'numeric_features_by_class.png'))
print("수치형 변수 분석 그래프 저장: numeric_features_by_class.png")


# =============================================================================
# [5] 데이터 전처리 (Data Preprocessing)
# =============================================================================
# 머신러닝 모델은 문자열을 이해하지 못하므로, 모든 데이터를 숫자로 변환해야 합니다.
# 범주형 데이터 -> Label Encoding 적용

print("\n" + "="*80)
print("4. 데이터 전처리")
print("="*80)

# 데이터 타입이 'object'(문자열)인 컬럼만 추출
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# 타겟 변수('Class')는 별도로 처리하기 위해 리스트에서 제외
categorical_cols.remove('Class')

# 나중에 원래 값을 확인하기 위해 인코더를 저장할 딕셔너리
label_encoders = {}
df_encoded = df.copy() # 원본 데이터 보존을 위해 복사본 사용

# 각 범주형 컬럼에 대해 LabelEncoder 적용
for col in categorical_cols:
    le = LabelEncoder()
    # 문자열 -> 숫자 변환 (예: 'Male'->1, 'Female'->0)
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le # 인코더 저장
    print(f"{col}: {len(le.classes_)} 개의 고유값 -> 0~{len(le.classes_)-1}로 인코딩")

# 타겟 변수(Class) 인코딩 (L, M, H -> 0, 1, 2)
target_le = LabelEncoder()
df_encoded['Class'] = target_le.fit_transform(df['Class'])
print(f"\n타겟 변수 Class: {target_le.classes_} -> {list(range(len(target_le.classes_)))}")

# 독립 변수(X)와 종속 변수(y) 분리
X = df_encoded.drop('Class', axis=1) # Class를 제외한 모든 컬럼
y = df_encoded['Class']              # 예측하려는 Class 컬럼

# 학습(Train) 데이터와 테스트(Test) 데이터 분리하기 (80:20 비율)
# random_state=42: 매번 똑같이 나뉘도록 시드 고정
# stratify=y: 학습/테스트 데이터에서 타겟 변수의 비율(L:M:H)을 원본과 동일하게 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
# 데이터가 균등하게 잘 나뉘었는지 확인
print(f"학습 데이터 타겟 분포:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"테스트 데이터 타겟 분포:\n{pd.Series(y_test).value_counts().sort_index()}")


# =============================================================================
# [6] 모델 학습 (Model Training)
# =============================================================================
# XGBoost Classifier를 사용하여 모델을 학습시킵니다.
# XGBoost: 여러 개의 결정 트리(Decision Tree)를 순차적으로 만들어 오차를 줄여나가는 앙상블 기법

print("\n" + "="*80)
print("5. XGBoost 모델 학습")
print("="*80)

# 모델 객체 생성 및 하이퍼파라미터 설정
xgb_model = XGBClassifier(
    n_estimators=100,    # 생성할 트리의 개수 (너무 많으면 과적합 가능성)
    max_depth=6,         # 트리의 최대 깊이 (복잡도 조절)
    learning_rate=0.1,   # 학습률 (각 트리가 학습에 미치는 영향 정도)
    random_state=42,     # 결과 재현성을 위한 난수 시드
    eval_metric='mlogloss' # 다중 분류를 위한 평가 지표 (로그 손실)
)

# 학습 데이터(X_train, y_train)로 모델 학습 수행
xgb_model.fit(X_train, y_train)
print("XGBoost 모델 학습 완료")


# =============================================================================
# [7] 모델 예측 및 평가 (Prediction & Evaluation)
# =============================================================================
# 학습된 모델이 테스트 데이터를 얼마나 잘 맞추는지 평가합니다.

print("\n" + "="*80)
print("6. 모델 평가")
print("="*80)

# 테스트 데이터에 대한 예측값 산출 (0, 1, 2 클래스 중 하나)
y_pred = xgb_model.predict(X_test)
# 테스트 데이터에 대한 각 클래스별 확률값 산출 (ROC AUC 계산용)
y_pred_proba = xgb_model.predict_proba(X_test)

# 1. 정확도(Accuracy) 계산: 전체 중 몇 개나 맞췄는지
accuracy = accuracy_score(y_test, y_pred)
# 정밀도(Precision): 모델이 Positive라고 예측한 것 중 실제 Positive인 비율
# 다중 클래스이므로 average='macro' (클래스별 평균) 사용
precision = precision_score(y_test, y_pred, average='macro')
# 재현율(Recall): 실제 Positive인 것 중 모델이 Positive라고 예측한 비율
recall = recall_score(y_test, y_pred, average='macro')
# F1 점수: 정밀도와 재현율의 조화 평균
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 2. 오차 행렬(Confusion Matrix): 어떤 클래스를 무엇으로 잘못 예측했는지 상세 확인
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# 3. 분류 리포트: 정밀도, 재현율, F1 점수를 클래스별로 보여줌
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# 4. ROC-AUC 점수: 모델의 변별력 평가 (1에 가까울수록 좋음)
# 다중 클래스이므로 'ovr'(One-vs-Rest) 방식 사용
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"\nROC-AUC Score (OvR): {roc_auc:.4f}")


# =============================================================================
# [8] 결과 시각화 (Results Visualization)
# =============================================================================
# 모델의 예측 결과와 변수 중요도를 시각화합니다.

print("\n" + "="*80)
print("7. 결과 시각화")
print("="*80)

# 1. Confusion Matrix 히트맵
# 숫자로만 보는 것보다 색상으로 표현하면 오분류 패턴을 파악하기 쉽습니다.
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_le.classes_, 
            yticklabels=target_le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted (예측값)')
plt.ylabel('Actual (실제값)')
plt.tight_layout()
plt.savefig(os.path.join(path, 'confusion_matrix.png'))
print("Confusion Matrix 저장: confusion_matrix.png")

# 2. ROC Curve (다중 클래스 OvR 방식)
# 각 클래스별로 ROC 곡선을 그립니다.
plt.figure(figsize=(10, 8))
n_classes = len(target_le.classes_)

# 성적 등급 한글 매핑 (H/L/M -> 우수/부진/보통)
class_kor_names = {'H': '우수(H)', 'L': '부진(L)', 'M': '보통(M)'}

for i in range(n_classes):
    # 각 클래스 i에 대한 FPR, TPR 계산
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
    class_name = target_le.classes_[i]
    plt.plot(fpr, tpr, label=f'{class_kor_names[class_name]} 등급 (AUC = {roc_auc_score(y_test == i, y_pred_proba[:, i]):.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='무작위 예측')
plt.xlabel('허위 양성 비율 (FPR)')
plt.ylabel('진 진짜 양성 비율 (TPR)')
plt.title('다중 클래스 ROC 곡선 (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(path, 'roc_curve.png'))
print("ROC Curve 그래프 저장: roc_curve.png")

# 3. Feature Importance (변수 중요도)
# 모델이 성적을 예측할 때 어떤 변수가 가장 중요한 역할을 했는지 확인합니다.
# 변수 이름을 한글로 변환하여 저장
korean_features = [feature_kor_names.get(col, col) for col in X.columns]

feature_importance = pd.DataFrame({
    'feature': korean_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False) # 중요도 내림차순 정렬

print(f"\n주요 Feature Importance (Top 10):")
print(feature_importance.head(10))

# 중요도 상위 10개 변수에 대해 가로 막대 그래프 그리기
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
# barh: 가로 막대 그래프
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('중요도 (Importance)')
plt.title('모델 예측 변수 중요도 Top 10 (XGBoost)')
plt.gca().invert_yaxis() # 중요도가 높은 것이 위로 오도록 Y축 반전
plt.tight_layout()
plt.savefig(os.path.join(path, 'feature_importance.png'))
print("Feature Importance 그래프 저장: feature_importance.png")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
