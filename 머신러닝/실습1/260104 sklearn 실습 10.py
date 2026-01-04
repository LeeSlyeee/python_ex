# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석 및 머신러닝 모델링에 필요한 핵심 라이브러리들을 불러옵니다.

# 데이터 조작 및 분석을 위한 라이브러리
# pandas: 데이터프레임(표 형태의 데이터)을 다루기 위해 사용 (pd라는 별칭 사용)
import pandas as pd
# numpy: 수치 계산, 배열 처리를 위해 사용 (np라는 별칭 사용)
import numpy as np

# 데이터 시각화를 위한 라이브러리
# matplotlib.pyplot: 기본적인 그래프 그리기 도구 (plt라는 별칭 사용)
import matplotlib.pyplot as plt
# seaborn: matplotlib 기반의 더 예쁘고 다양한 통계 차트를 그리기 위한 도구
import seaborn as sns

# Scikit-learn(사이킷런) 라이브러리 모듈 임포트
# 데이터셋 로드: OpenML에서 데이터셋을 가져오는 기능
from sklearn.datasets import fetch_openml
# 데이터 분할: 전체 데이터를 학습용(Train)과 테스트용(Test)으로 나누는 기능
from sklearn.model_selection import train_test_split
# 성능 평가 지표: 평균 절대 오차(MAE), 평균 제곱 오차(MSE) 계산 함수
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 회귀 모델들: 기본 선형 회귀, 릿지, 라쏘, 엘라스틱넷 등
from sklearn.linear_model import LinearRegression # 선형 회귀
from sklearn.linear_model import Ridge            # 릿지 회귀 (L2 규제)
from sklearn.model_selection import cross_val_score # 교차 검증 (사용되지 않지만 임포트됨)
from sklearn.linear_model import Lasso            # 라쏘 회귀 (L1 규제)
from sklearn.linear_model import ElasticNet       # 엘라스틱넷 (L1+L2 규제)
# 데이터 전처리(스케일링): 데이터의 범위를 조절하는 도구들
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# 파이프라인: 전처리 과정과 모델 학습 과정을 하나로 묶어주는 도구
from sklearn.pipeline import make_pipeline
# 특성 공학: 다항 특성(Polynomial Features)을 생성하여 모델의 복잡도를 높이는 도구
from sklearn.preprocessing import PolynomialFeatures


# =============================================================================
# [2] 데이터 로드 및 확인 (Data Loading & Inspection)
# =============================================================================

# Numpy 출력 옵션 설정: 소수점이 과학적 표기법(예: 1e-4)으로 나오지 않도록 설정
np.set_printoptions(suppress=True)

# OpenML에서 'boston' 데이터셋을 다운로드합니다. (version=1, 데이터프레임 형식으로)
# 보스턴 주택 가격 데이터셋은 주택 가격 예측 실습에 널리 사용되는 예제입니다.
data = fetch_openml(name="boston", version=1, as_frame=True)

# 데이터셋 객체에 어떤 키(정보)들이 들어있는지 확인합니다.
print(data.keys())
# 데이터셋에 대한 상세 설명(Description)을 출력하여 데이터의 의미를 파악합니다.
print(data['DESCR'])


# data['data']에는 특징(feature) 값들이 들어있습니다. 이를 pandas 데이터프레임으로 변환합니다.
df = pd.DataFrame(data['data'], columns=data['feature_names'])
# data['target']에는 우리가 예측해야 할 집값(MEDV)이 들어있습니다. 이를 데이터프레임에 추가합니다.
df['MEDV'] = data['target']
# 데이터프레임의 상위 5개 행을 출력하여 데이터가 잘 로드되었는지 눈으로 확인합니다.
print(df.head())


# 참고: as_frame=True 옵션을 썼기 때문에 data.frame 속성으로 전체 데이터프레임을 바로 얻을 수도 있습니다.
df2 = data.frame
# df2의 상위 5개 행 확인
df2.head()

# =============================================================================
# [3] 데이터셋 분리 (Train/Test Split)
# =============================================================================
# 모델 학습을 위해 전체 데이터를 학습 세트(Train Set)와 테스트 세트(Test Set)로 나눕니다.
# X: 특징 데이터 (.drop으로 타겟 컬럼인 'MEDV' 제외)
# y: 타겟 데이터 ('MEDV' 컬럼)
# 기본적으로 75:25 비율로 나뉩니다.
x_train, x_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1), df['MEDV'])

# 분리된 데이터의 크기(shape)를 확인합니다. (행 개수, 열 개수)
print(x_train.shape, x_test.shape)
# 학습용 입력 데이터(x_train) 확인
print(x_train.head())
# 학습용 정답 데이터(y_train) 확인
print(y_train.head())


# =============================================================================
# [4] 사용자 정의 평가 지표 함수 (Custom Evaluation Metrics)
# =============================================================================
# 모델 성능 평가 원리를 이해하기 위해 직접 MSE, MAE, RMSE 함수를 구현해봅니다.

# 테스트용 간단한 배열 생성
pred = np.array([3, 4, 5])   # 예측값
actual = np.array([1, 2, 3]) # 실제값

# 1. 평균 제곱 오차 (MSE: Mean Squared Error)
# 오차(예측값 - 실제값)를 제곱한 뒤 평균을 냅니다. 큰 오차에 패널티를 줍니다.
def my_mse(pred, actual):
    return ((pred - actual)**2).mean()

# 테스트 데이터로 MSE 계산 및 출력
print(my_mse(pred, actual))

# 2. 평균 절대 오차 (MAE: Mean Absolute Error)
# 오차의 절댓값을 취한 뒤 평균을 냅니다. 직관적인 오차 크기를 알 수 있습니다.
def my_mae(pred, actual):
    return np.abs(pred - actual).mean()

# 테스트 데이터로 MAE 계산 및 출력
print(my_mae(pred, actual))


# 3. 평균 제곱근 오차 (RMSE: Root Mean Squared Error)
# MSE에 제곱근(Root)을 씌위 원래 데이터의 단위와 맞춥니다.
def my_rmse(pred, actual):
    return np.sqrt(my_mse(pred, actual))

# 테스트 데이터로 RMSE 계산 및 출력
print(my_rmse(pred, actual))


# 사이킷런(sklearn)에서 제공하는 함수와 내가 만든 함수의 결과가 같은지 비교 검증합니다.
print(my_mae(pred, actual), mean_absolute_error(pred, actual))
print(my_mse(pred, actual), mean_squared_error(pred, actual))


# =============================================================================
# [5] 시각화 및 통합 평가 유틸리티 (Visualization & Utility)
# =============================================================================

# 여러 모델의 성능(MSE)을 저장하여 나중에 비교하기 위한 딕셔너리
my_predictions = {}

# 그래프에 사용할 색상 목록
colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
            'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
            'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
            'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
            'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
            'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]

# 예측값과 실제값의 차이를 산점도(Scatter Plot)로 보여주는 함수
def plot_predictions(name_, pred, actual):
    # 시각화를 위해 데이터프레임 생성
    df = pd.DataFrame({'prediction': pred, 'actual': actual})
    # 실제값을 기준으로 오름차순 정렬 (그래프를 보기 편하게 만들기 위함)
    df = df.sort_values(by='actual').reset_index(drop=True)
    
    plt.figure(figsize=(12, 9))
    # 예측값은 빨간색 'x'로 표시
    plt.scatter(df.index, df['prediction'], marker='x', color='r')
    # 실제값은 검은색 원('o')으로 표시
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
    
    plt.title(name_, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)
    plt.show()


# 모델의 MSE를 계산하고, 산점도를 그리며, 현재까지 저장된 모든 모델의 성능을 비교 그래프로 출력하는 핵심 함수
def mse_eval(name_, pred, actual):
    global predictions # (참고: 코드 상에서 실제로 사용되지는 않는 전역 변수)
    global colors
    
    # 1. 예측값 vs 실제값 산점도 그리기
    plot_predictions(name_, pred, actual)
    
    # 2. MSE 계산 및 저장
    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse
    
    # 3. 모델 성능 순위표 출력 (MSE가 작을수록 좋음 - 여기서는 정렬만 함)
    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(y_value, columns=['model', 'mse'])
    print(df)
    
    # 4. 모델 간 성능 비교 바 차트(Bar Chart) 그리기
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    length = len(df)
    
    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['mse'])
    
    # 각 막대마다 랜덤 색상 적용 및 수치 텍스트 표시
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')
    
    plt.title('MSE Error', fontsize=18)
    plt.xlim(min_, max_)
    plt.show()


# 성능 기록 딕셔너리에서 특정 모델을 삭제하는 도구 함수
def remove_model(name_):
    global my_predictions
    try:
        del my_predictions[name_]
    except KeyError:
        return False
    return True


# =============================================================================
# [6] 기본 선형 회귀 (Baseline: Linear Regression)
# =============================================================================

# 선형 회귀 모델 생성 (n_jobs=-1: 가능한 모든 프로세서 코어 사용)
model = LinearRegression(n_jobs=-1)

# 테스트 데이터 타입 확인
print(x_test.dtypes)

# 데이터 타입을 float(실수)으로 명시적 변환 (혹시 모를 에러 방지)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# 모델 학습 (fit)
print(model.fit(x_train, y_train))
# 예측 수행 (predict)
pred = model.predict(x_test)
# 성능 평가 및 시각화 (mse_eval 사용) -> 'LinearRegression'이라는 이름으로 결과 저장
print(mse_eval('LinearRegression', pred, y_test))


# =============================================================================
# [7] 회귀 계수 시각화 유틸리티 (Coefficient Visualization Utility)
# =============================================================================
# 모델이 어떤 특징(Feature)을 중요하게 생각하는지(가중치가 큰지) 확인하기 위한 함수
def plot_coef(columns, coef):
    # 특징 이름과 계수 값을 매핑하여 데이터프레임 생성
    coef_df = pd.DataFrame(list(zip(columns, coef)))
    coef_df.columns=['feature', 'coef']
    # 계수 값 기준으로 내림차순 정렬
    coef_df = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)
    
    # 바 차트로 시각화
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.arange(len(coef_df)), coef_df['coef'])
    idx = np.arange(len(coef_df))
    ax.set_yticks(idx)
    ax.set_yticklabels(coef_df['feature'])
    fig.tight_layout()
    plt.show()


# =============================================================================
# [8] 라쏘 회귀 (Lasso Regression) - L1 규제
# =============================================================================
# Feature Selection 효과가 있는 라쏘 회귀를 다양한 규제 강도(alpha)로 실험합니다.

# 실험할 alpha 값 리스트 (값이 클수록 규제가 강함)
alphas = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]

# 각 alpha 값에 대해 모델 학습 및 평가 반복
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    pred = lasso.predict(x_test)
    mse_eval('Lasso(alpha={})'.format(alpha), pred, y_test)


# 규제 강도에 따른 계수 변화를 비교해보기 위한 상세 분석
# 매우 강한 규제 (alpha=100) -> 대부분의 계수가 0이 됨
lasso_100 = Lasso(alpha=100)
lasso_100.fit(x_train, y_train)
lasso_pred_100 = lasso_100.predict(x_test)

# 매우 약한 규제 (alpha=0.001) -> 선형 회귀와 비슷해짐
lasso_001 = Lasso(alpha=0.001)
lasso_001.fit(x_train, y_train)
lasso_pred_001 = lasso_001.predict(x_test)

# alpha=100일 때의 회귀 계수 시각화 및 출력
plot_coef(x_train.columns, lasso_100.coef_)
print(lasso_100.coef_)

# alpha=0.001일 때의 회귀 계수 시각화
plot_coef(x_train.columns, lasso_001.coef_)


# =============================================================================
# [9] 엘라스틱넷 (ElasticNet) - L1 + L2 규제 혼합
# =============================================================================
# 라쏘(L1)와 릿지(L2)를 섞어놓은 모델입니다. l1_ratio로 비율을 조절합니다.

# l1_ratio 변동 실험 (0에 가까우면 릿지, 1에 가까우면 라쏘)
ratios = [0.2, 0.5, 0.8]

for ratio in ratios:
    # alpha는 0.5로 고정하고 ratio만 변경
    elasticnet = ElasticNet(alpha=0.5, l1_ratio=ratio)
    elasticnet.fit(x_train, y_train)
    pred = elasticnet.predict(x_test)
    mse_eval('ElasticNet(l1_ratio={})'.format(ratio), pred, y_test)


# 규제 강도(alpha)를 5로 높이고, l1_ratio에 따른 차이 비교
# l1_ratio = 0.2 (L2 비중이 높음)
elsticnet_20 = ElasticNet(alpha=5, l1_ratio=0.2)
elsticnet_20.fit(x_train, y_train)
elasticnet_pred_20 = elsticnet_20.predict(x_test)

# l1_ratio = 0.8 (L1 비중이 높음)
elsticnet_80 = ElasticNet(alpha=5, l1_ratio=0.8)
elsticnet_80.fit(x_train, y_train)
elasticnet_pred_80 = elsticnet_80.predict(x_test)

# 각각의 계수 분포 시각화
plot_coef(x_train.columns, elsticnet_20.coef_)
plot_coef(x_train.columns, elsticnet_80.coef_)
# l1_ratio=0.8일 때의 계수 값 직접 출력
print(elsticnet_80.coef_)


# =============================================================================
# [10] 데이터 스케일링 (Data Scaling)
# =============================================================================
# 특성(Feature)들의 단위가 다르면 모델 학습에 악영향을 줄 수 있으므로 스케일링을 수행합니다.

# 원본 데이터 통계 확인
print(x_train.describe())

# 1. StandardScaler: 평균 0, 표준편차 1로 변환 (표준화)
std_scaler = StandardScaler()
std_scaled = std_scaler.fit_transform(x_train) # 학습 및 변환 동시에 수행
print(round(pd.DataFrame(std_scaled).describe(), 2)) # 결과 확인

# 2. MinMaxScaler: 최소값 0, 최대값 1로 변환 (사이값은 비율로 축소)
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(x_train)
print(round(pd.DataFrame(minmax_scaled).describe(), 2))

# 3. RobustScaler: 중앙값(Median)과 IQR(사분위수 범위) 사용. 이상치(Outlier)에 강함.
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(x_train)
print(round(pd.DataFrame(robust_scaled).median(), 2))


# =============================================================================
# [11] 파이프라인 (Pipeline) 활용
# =============================================================================
# 전처리(스케일러)와 모델 학습을 하나의 파이프라인으로 연결하여 코드를 간결하게 하고 데이터 누수를 방지합니다.

# 파이프라인 생성: [스케일러(StandardScaler) -> 모델(ElasticNet)]
elasticnet_pipeline = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.2))

# 파이프라인을 학습시키면 내부적으로 스케일링 후 모델 학습이 진행됩니다.
elasticnet_pred = elasticnet_pipeline.fit(x_train, y_train).predict(x_test)

# 파이프라인 적용 모델의 성능 평가 ('Standard ElasticNet')
print(mse_eval('Standard ElasticNet', elasticnet_pred, y_test))


# 비교를 위해 파이프라인(스케일링) 없이 동일한 하이퍼파라미터 모델 학습
elasticnet_no_pipeline = ElasticNet(alpha=0.1, l1_ratio=0.2)
no_pipeline_pred = elasticnet_no_pipeline.fit(x_train, y_train).predict(x_test)

# 스케일링 없는 모델 성능 평가 ('No Standard ElasticNet')
# 일반적으로 스케일링을 한 경우(위)가 성능이 더 좋습니다.
print(mse_eval('No Standard ElasticNet', elasticnet_pred, y_test))


# =============================================================================
# [12] 다항 특성 추가 (Polynomial Features)
# =============================================================================
# 특성들을 서로 곱하거나 제곱하여 새로운 특성을 만들어내, 비선형적인 관계를 모델링합니다.

# degree=2: 2차항까지 생성 (x1^2, x1*x2 등), include_bias=False: 상수항(1) 제외
poly = PolynomialFeatures(degree=2, include_bias=False)

# x_train 데이터의 첫 번째 행만 변환해서 예시로 확인
poly_features = poly.fit_transform(x_train)[0]
# 변환된 특성(2차항 포함) 확인
print(poly_features)
# 원본 특성 확인 (비교용)
print(x_train.iloc[0])


# 파이프라인 생성: [다항 특성 추가 -> 스케일러 -> 모델]
# 1. PolynomialFeatures로 특성 개수 늘리기
# 2. StandardScaler로 스케일링
# 3. ElasticNet으로 회귀 분석
poly_pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.2))

# 학습 및 예측
poly_pred = poly_pipeline.fit(x_train, y_train).predict(x_test)

# 다항 회귀 모델 성능 평가 ('Poly ElasticNet')
print(mse_eval('Poly ElasticNet', poly_pred, y_test))