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

# test_size=0.3: 전체 데이터의 30%를 테스트 데이터로 분리합니다.
# random_state=121: 난수 시드를 고정하여 항상 동일하게 데이터가 섞이도록 합니다.
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris.target, test_size=0.3, random_state=121)

# 학습 데이터(X_train, y_train)로만 학습을 수행합니다.
dt_clf.fit(X_train, y_train)

# 테스트 데이터(X_test)로 예측을 수행하고 정확도를 확인합니다.
pred = dt_clf.predict(X_test)
print(f'예측 정확도: {accuracy_score(y_test, pred):.4f}')


# ==========================================
# 3. DataFrame을 이용한 데이터 분할 실습
# ==========================================

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df.head())

# 피처 데이터(마지막 열 제외)와 타겟 데이터(마지막 열)를 분리합니다.
ftr_df = iris_df.iloc[:, :-1]
tgt_df = iris_df.iloc[:, -1]

# DataFrame 형태의 데이터도 train_test_split으로 분리가 가능합니다.
X_train, X_test, y_train, y_test = train_test_split(ftr_df, tgt_df, test_size=0.3, random_state=121)

print(type(X_train), type(X_test), type(y_train), type(y_test))

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

# KFold 객체의 split()을 호출하면 학습용/검증용 데이터의 인덱스를 반환합니다.
for train_index, test_index in KFold().split(features):
    # 인덱스를 이용하여 학습용(Train), 검증용(Test) 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # 학습 및 예측
    # 여기서는 매 반복마다 dt_clf를 재학습시킵니다.
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    n_iter += 1

    # 정확도 계산
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'#{n_iter} 정확도: {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}')
    print(f'# {n_iter} 검증 세트 인덱스: {n_iter} {test_index}')
    cv_accuracy.append(accuracy)

# 5개 폴드의 평균 정확도를 계산합니다.
print('\n ## 평균 검증 정확도: ', np.mean(cv_accuracy))


# ==========================================
# 5. K-Fold의 문제점 확인 (데이터 불균형 문제)
# ==========================================

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# 원본 데이터의 레이블 분포 확인 (각 50개씩 균등함)
print(iris_df['label'].value_counts())

# 3개의 폴드로 나눌 때, 레이블이 균등하게 섞이지 않는 현상을 확인합니다.
kfold = KFold(n_splits=3)
n_iter = 0

for train_index, test_index in kfold.split(features):
    n_iter += 1
    
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    
    # 학습/검증 레이블이 골고루 섞이지 않아서 정확하게 학습/검증하기 어려움을 보여줍니다.
    print(f'## 교차 검증 {n_iter}')
    print(f'학습 레이블 빈도수: \n{label_train.value_counts()}')
    print(f'검증 레이블 빈도수: \n{label_test.value_counts()}')


# ==========================================
# 6. Stratified K-Fold (불균형한 분포를 가진 데이터에 적합)
# ==========================================

# StratifiedKFold는 원본 데이터의 레이블 분포를 고려하여, 이와 동일하게 학습/검증 세트를 분배합니다.
skf = StratifiedKFold(n_splits=3)
n_iter = 0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    
    # 위와 달리 레이블이 균형 잡히게 분포되었음을 확인할 수 있습니다.
    print(f'## 교차 검증 {n_iter}')
    print(f'학습 레이블 빈도수: \n{label_train.value_counts()}')
    print(f'검증 레이블 빈도수: \n{label_test.value_counts()}')


# ==========================================
# 7. Stratified K-Fold를 이용한 교차 검증 수행
# ==========================================

dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuracy = []

# split() 호출 시 반드시 레이블 데이터셋(label)도 함께 넣어주어야 합니다.
for train_index, test_index in skfold.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    n_iter += 1
    
    # 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f'#{n_iter} 정확도: {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}')
    print(f'# {n_iter} 검증 세트 인덱스: {n_iter} {test_index}')
    cv_accuracy.append(accuracy)

print(f'## 교차 검증별 정확도 {np.round(cv_accuracy, 4)}')
print(f'## 평균 검증 정확도: {np.mean(cv_accuracy):.4f}')


# ==========================================
# 8. cross_val_score (교차 검증을 간편하게 수행하는 API)
# ==========================================
    
iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
# iris_data.target을 사용하여 레이블 데이터를 가져옵니다.
label = iris_data.target   

# cross_val_score: 내부적으로 Stratified K-Fold를 사용하여 교차 검증을 수행하고 점수를 반환합니다.
# cv=3 : 3개의 폴드로 나눔. scoring='accuracy' : 평가지표로 정확도를 사용.
scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)

# cross_validate: cross_val_score와 유사하지만, 여러 평가지표를 동시에 반환할 수 있고, 수행 시간 정보도 제공합니다.
# scoring=['accuracy','recall','precision','f1']: 정확도, 재현율, 정밀도, F1 스코어를 모두 계산합니다.
# (무엇보다 실행 시 에러가 날 수 있는데, 이진 분류/다중 분류 설정에 따라 average 옵션이 필요할 수 있습니다.)
# 반환값은 딕셔너리 형태입니다. (fit_time, score_time, test_accuracy, test_recall, ...)
# 참고: Iris 데이터셋은 다중 분류(3개 클래스)이므로 recall, precision, f1 계산 시 average='macro' 또는 'weighted' 등의 추가 설정이 필요하여 경고나 에러가 발생할 수 있습니다.
scores2 = cross_validate(dt_clf, data, label, scoring=['accuracy','recall','precision','f1'], cv=3)

print("CV Keys (반환된 딕셔너리 키):", scores2.keys())

print(f'교차 검증별 정확도: {np.round(scores, 4)}')
print(f'평균 검증 정확도: {np.round(np.mean(scores), 4)}')


# ==========================================
# 9. GridSearchCV (교차 검증과 하이퍼파라미터 튜닝을 동시에)
# ==========================================

iris = load_iris()
# 테스트 데이터를 별도로 분리해 둡니다.
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)

dtree = DecisionTreeClassifier()

# 파라미터 딕셔너리 설정
# max_depth: 트리의 최대 깊이, min_samples_split: 노드를 분할하기 위한 최소 샘플 수
parameters = {'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}

# GridSearchCV 객체 생성
# param_grid: 튜닝할 파라미터들
# cv=3: 3-Fold 교차 검증 수행
# refit=True: 최적의 파라미터를 찾으면, 그 파라미터로 전체 학습 데이터에 대해 다시 학습(refit)을 수행
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)

# 붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과는 cv_results_ 딕셔너리에 저장됩니다. DataFrame으로 변환하여 확인.
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']])

# 전체 파라미터별 상세 결과 확인
print(grid_dtree.cv_results_)

# 최적의 파라미터와 그때의 정확도 확인
print(f'GridSearchCV 최적 파라미터: {grid_dtree.best_params_}')
print(f'GridSearchCV 최적 정확도: {grid_dtree.best_score_:.4f}')


pred = grid_dtree.predict(X_test)
print(f'테스트 데이터 세트 정확도: {accuracy_score(y_test, pred):.4f}')


estimator = grid_dtree.best_estimator_
pred = estimator.predict(X_test)
print(f'테스트 데이터 세트 정확도: {accuracy_score(y_test, pred):.4f}')




