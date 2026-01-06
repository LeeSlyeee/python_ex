# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 머신러닝 실습에 필요한 다양한 라이브러리들을 모듈별로 불러옵니다.

# 1. 시각화 및 데이터 핸들링 라이브러리
import matplotlib.pyplot as plt  # 그래프 그리기
import seaborn as sns            # 예쁜 통계 차트 그리
import pandas as pd              # 데이터프레임(표) 조작
import numpy as np               # 수치 연산 및 배열 처리
import requests                  # 인터넷에서 이미지 등 파일 다운로드
from io import BytesIO           # 다운로드 받은 바이너리 데이터를 메모리 상에서 파일처럼 취급
from PIL import Image            # 이미지 파일 열기 및 처리 (PIL 라이브러리 필요)

# 2. 사이킷런(Scikit-learn) - 데이터셋 및 기본 도구
from sklearn.datasets import fetch_openml  # OpenML 데이터셋 가져오기
from sklearn.model_selection import train_test_split # 학습/테스트 데이터 분리
from sklearn.metrics import mean_absolute_error, mean_squared_error # 평가 지표

# 3. 선형 회귀 계열 모델 (Linear Models)
from sklearn.linear_model import LinearRegression # 기본 선형 회귀
from sklearn.linear_model import Ridge            # 릿지 (L2 규제)
from sklearn.linear_model import Lasso            # 라쏘 (L1 규제)
from sklearn.linear_model import ElasticNet       # 엘라스틱넷 (L1+L2 규제)

# 4. 데이터 전처리 및 파이프라인
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # 스케일러
from sklearn.pipeline import make_pipeline      # 전처리+모델 파이프라인
from sklearn.preprocessing import PolynomialFeatures # 다항 특성 생성

# 5. 앙상블(Ensemble) 및 기타 모델
# Voting: 여러 모델의 결과를 투표나 평균으로 결합
from sklearn.ensemble import VotingRegressor 
from sklearn.ensemble import VotingClassifier
# 기타 분류/회귀 모델 (분류기인 Classifier는 이번 회귀 실습에 직접 쓰이진 않지만 임포트됨)
from sklearn.linear_model import LogisticRegression, RidgeClassifier 
# RandomForest: 배깅(Bagging) 대표 모델
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Boosting 계열: 이전 모델의 오차를 줄여나가는 방식
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
# 외부 라이브러리 XGBoost (성능이 좋음)
from xgboost import XGBRegressor, XGBClassifier
# 외부 라이브러리 LightGBM (빠르고 성능이 좋음)
from lightgbm import LGBMRegressor, LGBMClassifier
# Stacking: 여러 모델의 예측값을 다시 학습 데이터로 사용하는 메타 모델링
from sklearn.ensemble import StackingRegressor

# 6. 하이퍼파라미터 튜닝 및 교차 검증
from sklearn.model_selection import KFold             # K-Fold 교차 검증
from sklearn.model_selection import RandomizedSearchCV # 무작위 탐색 튜닝
from sklearn.model_selection import GridSearchCV       # 격자 탐색 튜닝


# =============================================================================
# [2] 데이터 로드 및 전처리 (Data Loading & Preprocessing)
# =============================================================================

# Numpy 출력 옵션: 과학적 표기법(예: 1e-4) 대신 소수점으로 출력하여 가독성 확보
np.set_printoptions(suppress=True)

# OpenML에서 보스턴 집값 데이터셋 가져오기
data = fetch_openml(name="boston", version=1, as_frame=True)

# 데이터를 DataFrame으로 변환
df = pd.DataFrame(data['data'], columns=data['feature_names'])
# 타겟 변수(집값, MEDV) 추가
df['MEDV'] = data['target']
# 상위 5개 데이터 확인
print(df.head())


# 학습 데이터와 테스트 데이터 분리 (Train/Test Split)
# test_size를 지정하지 않으면 기본적으로 0.25 (25%)가 테스트 셋이 됨
# random_state=42: 실행할 때마다 똑같이 나뉘도록 시드 고정
x_train, x_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1), df['MEDV'], random_state=42)

# 잘 나뉘었는지 크기 확인
print(x_train.shape, x_test.shape)
print(x_train.head())
print(y_train.head())


# =============================================================================
# [3] 평가 및 시각화 도구 (Evaluation Dictionary & Functions)
# =============================================================================

# 모델들의 예측 오차(MSE)를 수집하여 비교하기 위한 딕셔너리
my_predictions = {}

# 그래프 색상 리스트 (다양한 색상 미리 정의)
colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
           'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
           'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
           'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
           'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
           'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]


# 예측값과 실제값의 차이를 산점도로 시각화하는 함수
def plot_predictions(name_, pred, actual):
    df = pd.DataFrame({'prediction': pred, 'actual': y_test})
    df = df.sort_values(by='actual').reset_index(drop=True) # 실제값 기준 정렬
    plt.figure(figsize=(12, 9))
    plt.scatter(df.index, df['prediction'], marker='x', color='r') # 예측값: 빨간 x
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black') # 실제값: 검은 원
    plt.title(name_, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)
    plt.show()


# 모델 성능 평가(MSE) 및 전체 모델 비교 시각화 함수 (가장 중요한 함수)
def mse_eval(name_, pred, actual):
    global predictions # (사용되지 않지만 선언됨)
    global colors
    
    # 1. 예측 분포 시각화
    plot_predictions(name_, pred, actual)
    
    # 2. MSE 계산 및 기록
    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse
    
    # 3. 리더보드(순위표) 출력 (오차 높은 순 -> 낮은 순 정렬 필요할 듯 하나, 코드엔 reverse=True로 되어있음. 즉, 오차 큰게 위로 옴)
    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(y_value, columns=['model', 'mse'])
    print(df)
    
    # 4. 수평 막대 그래프로 성능 비교
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    length = len(df)
    
    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['mse'])
    
    # 막대마다 랜덤 색상 및 수치 텍스트 표시
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')
    
    plt.title('MSE Error', fontsize=18)
    plt.xlim(min_, max_)
    plt.show()


# 등록된 모델 기록을 삭제하는 헬퍼 함수
def remove_model(name_):
    global my_predictions
    try:
        del my_predictions[name_]
    except KeyError:
        return False
    return True


# 회귀 계수(Feature Importance)를 막대 그래프로 시각화하는 함수
def plot_coef(columns, coef):
    coef_df = pd.DataFrame(list(zip(columns, coef)))
    coef_df.columns=['feature', 'coef']
    coef_df = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.arange(len(coef_df)), coef_df['coef'])
    idx = np.arange(len(coef_df))
    ax.set_yticks(idx)
    ax.set_yticklabels(coef_df['feature'])
    fig.tight_layout()
    plt.show()


# =============================================================================
# [4] 기본 선형 모델 학습 및 평가 (Linear Models)
# =============================================================================
# 다양한 규제(Regularization) 방식을 적용한 선형 회귀 모델들을 비교합니다.

# 데이터 타입 변환 (안전하게 float형으로)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# 1. LinearRegression (기본 선형 회귀)
linear_reg = LinearRegression(n_jobs=-1)
linear_reg.fit(x_train, y_train)
pred = linear_reg.predict(x_test)
print(mse_eval('LinearRegression', pred, y_test))


# 2. Ridge (L2 규제: 계수 값을 0에 가깝게 줄임)
ridge = Ridge(alpha=1)
ridge.fit(x_train, y_train)
pred = ridge.predict(x_test)
print(mse_eval('Ridge(alpha=1)', pred, y_test))


# 3. Lasso (L1 규제: 중요하지 않은 특성의 계수를 0으로 만듦)
lasso = Lasso(alpha=0.01)
lasso.fit(x_train, y_train)
pred = lasso.predict(x_test)
print(mse_eval('Lasso(alpha=0.01)', pred, y_test))


# 4. ElasticNet (L1 + L2 혼합 규제)
elasticnet = ElasticNet(alpha=0.5, l1_ratio=0.8)
elasticnet.fit(x_train, y_train)
pred = elasticnet.predict(x_test)
print(mse_eval('ElasticNet(l1_ratio=0.8)', pred, y_test))


# 5. Pipeline - StandardScaler + ElasticNet
# 데이터 스케일링을 모델 학습 과정에 포함시켜 전처리를 자동화
elasticnet_pipeline = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.2))
elasticnet_pred = elasticnet_pipeline.fit(x_train, y_train).predict(x_test)
print(mse_eval('Standard ElasticNet', elasticnet_pred, y_test))


# 6. Pipeline - PolynomialFeatures + StandardScaler + ElasticNet
# 특성을 다항으로 확장하여(비선형성 추가) 모델의 표현력을 높임
poly_pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.2))
poly_pred = poly_pipeline.fit(x_train, y_train).predict(x_test)
print(mse_eval('Poly ElasticNet', poly_pred, y_test))


# =============================================================================
# [5] 앙상블 (Ensemble) - Voting
# =============================================================================
# 여러 개의 단일 모델들을 결합하여 더 좋은 예측 성능을 내는 기법입니다.

# 투표에 참여할 모델 리스트 정의
single_models = [
    ('linear_reg', linear_reg),
    ('ridge', ridge),
    ('lasso', lasso),
    ('elasticnet_pipeline', elasticnet_pipeline),
    ('poly_pipeline', poly_pipeline)
]

# VotingRegressor: 각 모델의 예측값을 평균 내어 최종 예측 (Soft Voting과 유사)
voting_regressor = VotingRegressor(single_models, n_jobs=-1)
voting_regressor.fit(x_train, y_train)
voting_pred = voting_regressor.predict(x_test)
print(mse_eval('Voting Ensemble', voting_pred, y_test))


# 참고: VotingClassifier 예시 (회귀 문제라 실제 사용되진 않음)
models = [
    ('Logi', LogisticRegression()),
    ('ridge', RidgeClassifier())
]
vc = VotingClassifier(models, voting='hard') # Hard Voting: 다수결


# 이미지 자료: 앙상블 개념도 1
try:
    response = requests.get('https://teddylee777.github.io/images/2019-12-17/image-20191217015537872.png')
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
except:
    print("이미지 로드 실패 (무시 가능)")


# =============================================================================
# [6] 앙상블 (Ensemble) - Bagging (Random Forest)
# =============================================================================
# 여러 개의 결정 트리(Decision Tree)를 만들고 그 결과를 종합하는 방식입니다.

# 1. 기본 Random Forest
rfr = RandomForestRegressor()
print(rfr.fit(x_train, y_train))

rfr_pred = rfr.predict(x_test)
print(mse_eval('RandomForest Ensemble', rfr_pred, y_test))


# 이미지 자료: 결정 트리 예시
try:
    response = requests.get('https://teddylee777.github.io/images/2020-01-09/decistion-tree.png')
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
except:
    print("이미지 로드 실패")


# 2. 하이퍼파라미터 튜닝된 Random Forest
# n_estimators: 나무의 개수, max_depth: 나무의 깊이 등 제한을 두어 과적합 방지
rfr = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=7, max_features=0.9)
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)
print(mse_eval('RandomForest Ensemble w/ Tuning', rfr_pred, y_test))


# 이미지 자료: 부스팅 개념도
try:
    response = requests.get('https://keras.io/img/graph-kaggle-1.jpeg')
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
except:
    print("이미지 로드 실패")


# =============================================================================
# [7] 앙상블 (Ensemble) - Boosting (Gradient Boosting, XGBoost, LightGBM)
# =============================================================================
# 이전 모델이 틀린 오차를 다음 모델이 학습하여 점진적으로 성능을 높이는 방식입니다.

# 1. Gradient Boosting - 기본
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
print(mse_eval('GradientBoost Ensemble', gbr_pred, y_test))


# 2. Gradient Boosting - 학습률(learning_rate) 조정
# 학습률이 작으면 더 많은 트리가 필요하지만 성능이 정교해질 수 있음
gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.01)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
print(mse_eval('GradientBoost Ensemble (lr=0.01)', gbr_pred, y_test))


# 3. Gradient Boosting - 트리 개수(n_estimators) 늘리기
gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.01, n_estimators=1000)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
print(mse_eval('GradientBoost Ensemble (lr=0.01, est=1000)', gbr_pred, y_test))


# 4. Gradient Boosting - subsample 추가 (Stochastic Gradient Boosting)
# 전체 데이터가 아닌 일부 데이터만 샘플링하여 학습 (속도 향상 및 과적합 방지)
gbr = GradientBoostingRegressor(random_state=42, learning_rate=0.01, n_estimators=1000, subsample=0.8)
gbr.fit(x_train, y_train)
gbr_pred = gbr.predict(x_test)
print(mse_eval('GradientBoost Ensemble (lr=0.01, est=1000, subsample=0.8)', gbr_pred, y_test))


# 5. XGBoost (eXtreme Gradient Boosting) - 성능과 속도가 최적화된 라이브러리
xgb = XGBRegressor(random_state=42)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
print(mse_eval('XGBoost', xgb_pred, y_test))

# XGBoost 튜닝 모델
xgb = XGBRegressor(random_state=42, learning_rate=0.01, n_estimators=1000, subsample=0.8, max_features=0.8, max_depth=7)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
print(mse_eval('XGBoost w/ Tuning', xgb_pred, y_test))


# 6. LightGBM - 더 빠르고 메모리 효율적인 부스팅 모델 (Leaf-wise 성장)
lgbm = LGBMRegressor(random_state=42, verbose=-1) # verbose=-1: 경고 메시지 끄기
lgbm.fit(x_train, y_train)
lgbm_pred = lgbm.predict(x_test)
print(mse_eval('LGBM', lgbm_pred, y_test))

# LightGBM 튜닝 모델
lgbm = LGBMRegressor(random_state=42, learning_rate=0.01, n_estimators=2000, colsample_bytree=0.9, subsample=0.9, max_depth=7)
lgbm.fit(x_train, y_train)
lgbm_pred = lgbm.predict(x_test)
print(mse_eval('LGBM w/ Tuning', lgbm_pred, y_test))


# =============================================================================
# [8] 앙상블 (Ensemble) - Stacking & Weighted Blending
# =============================================================================
# 여러 모델의 예측 결과를 다시 학습 데이터로 사용하여 최종 메타 모델이 예측을 수행하는 고급 기법입니다.

# 1. Stacking Ensemble
# 1단계 모델(Base Learners)들이 예측한 값을 모아서
stack_models = [
    ('elasticnet', poly_pipeline),
    ('randomforest', rfr),
    ('gbr', gbr),
    ('lgbm', lgbm),
]

# 2단계 모델(Final Estimator)이 최종 학습 (여기서는 XGBoost 사용)
stack_reg = StackingRegressor(stack_models, final_estimator=xgb, n_jobs=-1)

stack_reg.fit(x_train, y_train)
stack_pred = stack_reg.predict(x_test)
print(mse_eval('Stacking Ensemble', stack_pred, y_test))


# 2. Weighted Blending (가중 평균)
# 단순히 평균을 내는 Voting과 달리, 성능 좋은 모델에 가중치를 더 주어 합칩니다.
final_outputs = {
    'elasticnet': poly_pred,
    'randomforest': rfr_pred,
    'gbr': gbr_pred,
    'xgb': xgb_pred,
    'lgbm': lgbm_pred,
    'stacking': stack_pred,
}

# 각 모델별 가중치 부여 (총합 1.0)
final_prediction=\
final_outputs['elasticnet'] * 0.1\
+final_outputs['randomforest'] * 0.1\
+final_outputs['gbr'] * 0.2\
+final_outputs['xgb'] * 0.25\
+final_outputs['lgbm'] * 0.15\
+final_outputs['stacking'] * 0.2

print(mse_eval('Weighted Blending', final_prediction, y_test))


# 이미지 로드 (스택킹 관련 이미지로 추정되나 URL 불명확, 예외처리 없이 원본 유지)
try:
    response = requests.get('https://static.packt-cdn.com/products/9781789617740/graphics/b04c27c5-7e3f-428a-9aa6-bb3ebcd3584c.png') # 아마 잘못된 URL일 가능성 높음
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
except:
    print("이미지 로드 실패")


# =============================================================================
# [9] 교차 검증 (Cross Validation)
# =============================================================================
# 데이터를 학습/검증 세트로 한 번만 나누는 것이 아니라, 여러 번 나누어(K-Fold) 모델의 일반화 성능을 검증합니다.

# 5개의 폴드로 나눔, 섞어서(shuffle=True)
n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
print(df.head())

# 데이터프레임을 Numpy 배열로 변환
X = np.array(df.drop('MEDV', axis=1))
Y = np.array(df['MEDV'])

lgbm_fold = LGBMRegressor(random_state=42)

i = 1
total_error = 0
# 각 폴드마다 학습 및 평가 수행
for train_index, test_index in kfold.split(X):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    
    # 학습 및 예측
    lgbm_pred_fold = lgbm_fold.fit(x_train_fold, y_train_fold).predict(x_test_fold)
    
    error = mean_squared_error(lgbm_pred_fold, y_test_fold)
    print('Fold = {}, prediction score = {:.2f}'.format(i, error))
    total_error += error
    i+=1

print('---'*10)
# 5번 실행한 오차값의 평균 출력
print('Average Error: %s' % (total_error / n_splits))


# =============================================================================
# [10] 하이퍼파라미터 튜닝 (Hyperparameter Tuning)
# =============================================================================
# 모델의 성능을 최대로 끌어올리기 위해 최적의 파라미터 조합을 찾습니다.

# 1. RandomizedSearchCV (무작위 탐색)
# 모든 조합을 다 해보지 않고 랜덤하게 몇 개(n_iter)만 뽑아서 테스트. 속도가 빠름.
params = {
    'n_estimators': [200, 500, 1000, 2000],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [6, 7, 8],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'subsample': [0.8, 0.9, 1.0],
}

clf = RandomizedSearchCV(LGBMRegressor(), params, random_state=42, cv=3, n_iter=25, scoring='neg_mean_squared_error')
clf.fit(x_train, y_train)
# 최적의 점수와 파라미터 출력
print(clf.best_score_)
print(clf.best_params_)


# 2. GridSearchCV (격자 탐색)
# 설정한 모든 파라미터 조합을 전수 조사. 정확하지만 시간이 오래 걸림.
params = {
    'n_estimators': [500, 1000],          # 2개
    'learning_rate': [0.1, 0.05, 0.01],   # 3개
    'max_depth': [7, 8],                  # 2개
    'colsample_bytree': [0.8, 0.9],       # 2개
    'subsample': [0.8, 0.9,],             # 2개
} # 총 2 * 3 * 2 * 2 * 2 = 48번 탐색 (거기에 cv=3이므로 144번 학습)

grid_search = GridSearchCV(LGBMRegressor(), params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)
# 최적의 점수와 파라미터 출력
print(grid_search.best_score_)
print(grid_search.best_params_)


# 3. 최적의 파라미터로 최종 모델 학습 및 평가
# GridSearchCV 등을 통해 찾은 최적의 파라미터를 사용
lgbm_best = LGBMRegressor(n_estimators=500, subsample=0.8, max_depth=7, learning_rate=0.05, colsample_bytree=0.8)
lgbm_best_pred = lgbm_best.fit(x_train, y_train).predict(x_test)
print(mse_eval('GridSearch LGBM', lgbm_best_pred, y_test))