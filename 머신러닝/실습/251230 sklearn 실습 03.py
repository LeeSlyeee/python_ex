import sklearn
from sklearn.datasets import load_iris  # 붓꽃(Iris) 데이터 세트를 불러오기 위한 함수
from sklearn.tree import DecisionTreeClassifier  # 의사결정트리(Decision Tree) 분류 알고리즘 클래스
from sklearn.metrics import accuracy_score  # 모델의 성능(정확도)을 평가하기 위한 함수
from sklearn.model_selection import train_test_split  # 학습 데이터와 테스트 데이터를 분리하는 함수
from sklearn.model_selection import KFold  # K-Fold 교차 검증을 위한 클래스
from sklearn.model_selection import StratifiedKFold  # 레이블 불균형을 해결한 K-Fold 클래스
from sklearn.model_selection import cross_val_score, cross_validate  # 교차 검증을 간편하게 수행하는 API
from sklearn.model_selection import GridSearchCV  # 교차 검증과 최적 하이퍼 파라미터 튜닝을 한 번에 수행하는 클래스
import pandas as pd  # 데이터를 표(DataFrame) 형태로 다루기 위한 라이브러리
import numpy as np  # 수치 연산 및 배열(ndarray) 처리를 위한 라이브러리

# ==========================================
# 1. 데이터 로드 및 학습 데이터로만 학습/예측 (과적합 문제 발생 예시)
# ==========================================

# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()
dt_clf = DecisionTreeClassifier()

# 데이터를 별도의 테스트 세트로 나누지 않고, 전체 데이터를 학습 데이터로 사용합니다.
# 이렇게 하면 모델이 이미 본 데이터로 시험을 치르는 것이므로 과적합(Overfitting)이 발생합니다.
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data, train_label)

# 학습된 데이터로 다시 예측을 수행합니다.
# 이미 답을 알고 있는 데이터로 시험을 보는 것과 같으므로 정확도가 1.0(100%)이 나옵니다.
pred = dt_clf.predict(train_data)
print('예측 정확도: ', accuracy_score(train_label, pred))


# ==========================================
# 2. train_test_split을 이용한 학습/테스트 데이터 분리
# ==========================================

dt_clf = DecisionTreeClassifier()
iris_data = iris.data

# test_size=0.3: 전체 데이터의 30%를 테스트 데이터로 분리합니다. (학습 70%, 테스트 30%)
# random_state=121: 난수 시드를 고정하여 코드를 실행할 때마다 항상 동일하게 데이터가 섞이도록 합니다.
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris.target, test_size=0.3, random_state=121)

# 분리된 학습 데이터(X_train, y_train)로만 학습을 수행합니다.
dt_clf.fit(X_train, y_train)

# 학습에 사용하지 않은 테스트 데이터(X_test)로 예측을 수행하고 정확도를 확인합니다.
pred = dt_clf.predict(X_test)
print(f'예측 정확도: {accuracy_score(y_test, pred):.4f}')


# ==========================================
# 3. DataFrame을 이용한 데이터 분할 실습
# ==========================================

# 붓꽃 데이터를 보기 좋게 DataFrame으로 변환합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df.head())

# 피처 데이터(마지막 'target' 열 제외)와 타겟 데이터(마지막 'target' 열)를 분리합니다.
ftr_df = iris_df.iloc[:, :-1]
tgt_df = iris_df.iloc[:, -1]

# DataFrame 형태의 데이터도 train_test_split으로 바로 분리가 가능합니다.
X_train, X_test, y_train, y_test = train_test_split(ftr_df, tgt_df, test_size=0.3, random_state=121)

print(type(X_train), type(X_test), type(y_train), type(y_test))

# DataFrame으로 분리된 데이터로 학습 및 예측을 수행합니다.
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print(f'예측 정확도: {accuracy_score(y_test, pred):.4f}')


# ==========================================
# 4. K-Fold 교차 검증 (K-Fold Cross Validation)
# ==========================================

iris = load_iris()
features = iris.data
label = iris.target

dt_clf = DecisionTreeClassifier(random_state=156)
cv_accuracy = []
print('붓꽃 데이터 세트 크기: ', features.shape[0])

# 5개의 폴드 세트로 분리하는 KFold 객체 생성
n_iter = 0

# KFold 객체의 split()을 호출하면 전체 데이터를 5등분하여 학습용/검증용 데이터의 인덱스를 반환합니다.
for train_index, test_index in KFold().split(features):
    # 반환된 인덱스를 이용하여 학습용(Train), 검증용(Test) 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # 학습 및 예측
    # 반복마다 모델을 초기화하지는 않았지만, fit()을 호출하면 새로운 데이터로 학습하게 됩니다.
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    n_iter += 1

    # 정확도 계산 및 출력
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'#{n_iter} 정확도: {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}')
    print(f'# {n_iter} 검증 세트 인덱스: {n_iter} {test_index}')
    cv_accuracy.append(accuracy)

# 5개 폴드의 평균 정확도를 계산하여 모델의 일반화 성능을 평가합니다.
print('\n ## 평균 검증 정확도: ', np.mean(cv_accuracy))


# ==========================================
# 5. K-Fold의 문제점 확인 (데이터 불균형 문제)
# ==========================================

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# 원본 데이터의 레이블 분포 확인 (0, 1, 2가 각 50개씩 균등함)
print(iris_df['label'].value_counts())

# 데이터를 순서대로 3등분 하는 일반 K-Fold의 경우, 특정 레이블이 학습/검증 세트에서 배제될 수 있습니다.
kfold = KFold(n_splits=3)
n_iter = 0

for train_index, test_index in kfold.split(features):
    n_iter += 1
    
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    
    # 학습 레이블과 검증 레이블의 분포를 출력하여 불균형을 확인합니다.
    # 예: 학습엔 0, 1만 있고 검증엔 2만 있는 경우 학습이 제대로 되지 않습니다.
    print(f'## 교차 검증 {n_iter}')
    print(f'학습 레이블 빈도수: \n{label_train.value_counts()}')
    print(f'검증 레이블 빈도수: \n{label_test.value_counts()}')


# ==========================================
# 6. Stratified K-Fold (불균형한 분포를 가진 데이터에 적합)
# ==========================================

# StratifiedKFold는 불균형한 분포를 가진 레이블(Target) 데이터 집합을 위한 K-Fold 방식입니다.
# 원본 데이터의 레이블 분포를 먼저 고려한 뒤, 이 분포와 동일하게 학습/검증 데이터 세트를 분배합니다.

dt_clf = DecisionTreeClassifier(random_state=156)

# 3개의 폴드로 나누며, Stratified 방식을 사용합니다.
skf = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuracy = []

# StratifiedKFold의 split() 호출 시 반드시 레이블 데이터셋(label)도 함께 넣어주어야 분포를 파악할 수 있습니다.
# (iris_df는 위 5번 예제에서 생성한 DataFrame을 그대로 사용)
for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    
    # 1. 데이터 분할
    # 데이터프레임에서 인덱스를 사용해 학습/검증 데이터를 나눕니다.
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # [검증] K-Fold와 달리 모든 폴드에서 레이블이 균일한 비율로 섞여 있음을 확인합니다.
    print(f'\n## Stratified 교차 검증 {n_iter}')
    print(f'학습 레이블 분포: \n{label_train.value_counts()}')
    print(f'검증 레이블 분포: \n{label_test.value_counts()}')
    
    # 2. 모델 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    # 3. 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    
    print(f'#{n_iter} 교차 검증 정확도: {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}')
    cv_accuracy.append(accuracy)

# 교차 검증별 정확도 및 평균 정확도 출력
print('\n## Stratified K-Fold 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print(f'## Stratified K-Fold 평균 검증 정확도: {np.mean(cv_accuracy):.4f}')


# ==========================================
# 7. cross_val_score (교차 검증을 간편하게 수행하는 API)
# ==========================================
    
iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target   

# cross_val_score: 위에서 for문으로 작성한 교차 검증 과정을 한 줄로 줄여줍니다.
# 특징 1: 내부적으로 Stratified K-Fold를 사용하여 레이블 분포를 유지합니다. (분류 모델인 경우)
# 특징 2: 교차 검증 후의 정확도(score)를 배열 형태로 반환합니다.
# cv=3 : 3개의 폴드로 나눔. scoring='accuracy' : 평가지표로 정확도를 사용.
scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)

print('\n[cross_val_score 수행 결과]')
print(f'교차 검증별 정확도: {np.round(scores, 4)}')
print(f'평균 검증 정확도: {np.round(np.mean(scores), 4)}')

# [보너스] cross_validate: 여러 개의 평가지표를 한 번에 확인하고 싶을 때 사용
# 반환값이 딕셔너리 형태라 {키: [값, 값, 값]} 구조를 가집니다.
scores2 = cross_validate(dt_clf, data, label, scoring='accuracy', cv=3)
print('\n[cross_validate 수행 결과 (딕셔너리 형태)]')
print("반환된 키 목록:", scores2.keys())
print("테스트 정확도(test_score):", np.round(scores2['test_score'], 4))


# ==========================================
# 8. GridSearchCV (교차 검증과 하이퍼파라미터 튜닝을 동시에)
# ==========================================

iris = load_iris()
# 테스트 데이터를 20% 별도로 분리해 둡니다. (나중에 최종 성능 평가용)
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)

dtree = DecisionTreeClassifier()

# 파라미터 딕셔너리 설정
# max_depth: 트리의 최대 깊이, min_samples_split: 노드를 분할하기 위한 최소 샘플 수
parameters = {'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}

# GridSearchCV 객체 생성
# param_grid: 튜닝할 파라미터들
# cv=3: 3-Fold 교차 검증 수행
# refit=True: 최적의 파라미터를 찾으면, 그 파라미터로 전체 학습 데이터(X_train)에 대해 다시 학습(refit)을 수행
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)

# 붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가
# 이 과정에서 모든 파라미터 조합에 대해 CV를 수행하고 가장 좋은 파라미터를 찾습니다.
grid_dtree.fit(X_train, y_train)

print('\n[GridSearchCV 수행 결과]')
# GridSearchCV 결과는 cv_results_ 딕셔너리에 저장됩니다. DataFrame으로 변환하여 확인.
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']])

# 최적의 파라미터와 그때의 정확도 확인
print(f'GridSearchCV 최적 파라미터: {grid_dtree.best_params_}')
print(f'GridSearchCV 최적 정확도: {grid_dtree.best_score_:.4f}')

# refit=True로 설정했으므로 grid_dtree 객체 자체가 이미 최적의 파라미터로 재학습된 Estimator 역할을 할 수 있습니다.
# 바로 predict()를 호출하면 내부적으로 최적의 모델을 사용하여 예측을 수행합니다.
pred = grid_dtree.predict(X_test)
print(f'테스트 데이터 세트 정확도 (grid_dtree 바로 사용): {accuracy_score(y_test, pred):.4f}')

# 또는 best_estimator_ 속성을 통해 최적의 모델(Estimator) 객체만 따로 꺼내와서 사용할 수도 있습니다.
estimator = grid_dtree.best_estimator_
pred = estimator.predict(X_test)
print(f'테스트 데이터 세트 정확도 (best_estimator_ 사용): {accuracy_score(y_test, pred):.4f}')
