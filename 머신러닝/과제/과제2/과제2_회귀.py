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
# 회귀 모델 평가 지표: 평균 제곱 오차, 결정 계수, 평균 절대 오차
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 회귀 알고리즘 라이브러리
from sklearn.linear_model import LinearRegression # 선형 회귀 (Baseline)
from xgboost import XGBRegressor                  # XGBoost 회귀 모델

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
print(f"데이터 크기: {df.shape}")
print(f"\n데이터 컬럼:\n{df.columns.tolist()}")
print(f"\n첫 5개 행:\n{df.head()}")
print(f"\n데이터 정보:\n{df.info()}")
print(f"\n기술 통계:\n{df.describe()}")
print(f"\n결측치:\n{df.isnull().sum()}")


# =============================================================================
# [3] 타겟 변수 전처리 및 탐색 (Target Preprocessing & EDA)
# =============================================================================
# 회귀 분석을 위해 성적 등급('Class')을 순서가 있는 숫자로 변환합니다.
# L (Low) -> 0
# M (Middle) -> 1
# H (High) -> 2

print("\n" + "="*80)
print("2. 타겟 변수 매핑 및 분포")
print("="*80)

# 클래스 매핑 정의
class_mapping = {'L': 0, 'M': 1, 'H': 2}

# 매핑 적용
df['Class_Num'] = df['Class'].map(class_mapping)

print("Class 매핑 결과 (상위 5개):")
print(df[['Class', 'Class_Num']].head())

# 타겟 변수 분포 시각화
plt.figure(figsize=(8, 6))
sns.countplot(x='Class_Num', data=df, palette='viridis')
plt.title('성적 등급(숫자 변환 후) 분포')
plt.xlabel('성적 점수 (0:Low, 1:Middle, 2:High)')
plt.ylabel('학생 수')
plt.xticks([0, 1, 2], ['Low (0)', 'Middle (1)', 'High (2)'])
plt.savefig(os.path.join(path, 'target_distribution_reg.png'))
print("타겟 분포 그래프 저장: target_distribution_reg.png")


# =============================================================================
# [4] 데이터 전처리 (Data Preprocessing)
# =============================================================================
# 나머지 범주형 데이터에 대해 Label Encoding을 적용합니다.

print("\n" + "="*80)
print("3. 데이터 전처리 (인코딩)")
print("="*80)

# 데이터 타입이 'object'(문자열)인 컬럼만 추출
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# 'Class'는 이미 숫자로 바꿨고, 원본 'Class'는 제외
if 'Class' in categorical_cols:
    categorical_cols.remove('Class')

label_encoders = {}
df_encoded = df.copy()

# 원본 Class 컬럼 제거 (이미 Class_Num으로 변환함)
df_encoded = df_encoded.drop('Class', axis=1)

# 범주형 컬럼 인코딩
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"{col}: 인코딩 완료")

# 독립 변수(X)와 종속 변수(y) 분리
X = df_encoded.drop('Class_Num', axis=1)
y = df_encoded['Class_Num']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")


# =============================================================================
# [5] 모델 학습 및 평가 (Model Training & Evaluation)
# =============================================================================
# 두 가지 회귀 모델을 학습하고 성능을 비교합니다.
# 1. Linear Regression (선형 회귀)
# 2. XGBoost Regressor

print("\n" + "="*80)
print("4. 회귀 모델 학습 및 평가")
print("="*80)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"[{model_name}] 성능 평가:")
    print(f"  MSE (Mean Squared Error): {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  R2 Score (결정 계수): {r2:.4f}")
    print("-" * 50)
    return y_pred, r2

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr, r2_lr = evaluate_model(lr_model, X_test, y_test, "Linear Regression")

# 2. XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb, r2_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost Regressor")


# =============================================================================
# [6] 결과 시각화 (Visualization)
# =============================================================================

print("\n" + "="*80)
print("5. 결과 시각화")
print("="*80)

# 1. 실제값 vs 예측값 비교 (XGBoost 기준)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.6, color='blue', label='예측값')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='완벽한 예측선')
plt.xlabel('실제값 (Actual Values)')
plt.ylabel('예측값 (Predicted Values)')
plt.title('실제 성적 vs 예측 성적 (XGBoost)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(path, 'actual_vs_predicted.png'))
print("실제값 vs 예측값 그래프 저장: actual_vs_predicted.png")

# 2. 예측값의 오차 분포 (Residual Plot)
residuals = y_test - y_pred_xgb
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.xlabel('잔차 (Residuals = 실제값 - 예측값)')
plt.title('예측 오차(잔차) 분포 확인')
plt.axvline(0, color='red', linestyle='--')
plt.grid(True)
plt.savefig(os.path.join(path, 'residuals_distribution.png'))
print("잔차 분포 그래프 저장: residuals_distribution.png")

# 3. Feature Importance (XGBoost)
feature_kor_names = {
    'raisedhands': '손을 들 횟수', 
    'VisITedResources': '과목 공지 확인 횟수',
    'AnnouncementsView': '공지사항 확인 횟수', 
    'Discussion': '토론 참여 횟수',
    'gender': '성별', 
    'NationalITy': '국적', 
    'PlaceofBirth': '태어난 국가',
    'StageID': '학교 단계', 
    'GradeID': '학년(ID)', 
    'SectionID': '반 이름',
    'Topic': '과목', 
    'Semester': '학기', 
    'Relation': '보호자 관계',
    'ParentAnsweringSurvey': '부모 설문 참여', 
    'ParentschoolSatisfaction': '부모 만족도',
    'StudentAbsenceDays': '결석 횟수'
}

korean_features = [feature_kor_names.get(col, col) for col in X.columns]
importance_df = pd.DataFrame({
    'Feature': korean_features,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('회귀 모델 중요 변수 Top 10 (XGBoost)')
plt.xlabel('중요도 (Importance)')
plt.ylabel('변수명')
plt.tight_layout()
plt.savefig(os.path.join(path, 'feature_importance_reg.png'))
print("변수 중요도 그래프 저장: feature_importance_reg.png")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
