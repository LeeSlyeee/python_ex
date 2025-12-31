from sklearn import preprocessing # 데이터 전처리를 위한 모듈 모음 (인코딩, 스케일링 등)
from sklearn.preprocessing import LabelEncoder # 범주형 문자 데이터를 숫자형으로 변환해주는 클래스
from sklearn.model_selection import train_test_split # 전체 데이터를 학습용과 테스트용으로 나누는 함수
from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 알고리즘을 구현한 분류 모델
from sklearn.ensemble import RandomForestClassifier # 앙상블 기법 중 하나인 랜덤 포레스트 분류 모델
from sklearn.linear_model import LogisticRegression # 분류를 위한 선형 모델인 로지스틱 회귀 모델
from sklearn.metrics import accuracy_score # 모델 성능 평가를 위한 정확도 계산 함수
from sklearn.model_selection import KFold # K-Fold 교차 검증을 수행하기 위한 객체
from sklearn.model_selection import cross_val_score # 교차 검증을 더 간편하게 수행할 수 있는 함수
from sklearn.model_selection import GridSearchCV # 하이퍼 파라미터 튜닝과 교차 검증을 동시에 수행하는 객체

import numpy as np # 선형대수 및 배열 연산을 위한 핵심 라이브러리
import pandas as pd # 표(DataFrame) 형태의 데이터 처리를 위한 강력한 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화(그래프)를 위한 기본 라이브러리
import seaborn as sns # matplotlib을 기반으로 더 예쁘고 통계적인 차트를 그리기 위한 라이브러리
import os # 파일 경로 조작 등 운영체제 기능을 사용하기 위한 모듈

# 현재 실행 중인 파이썬 스크립트 파일의 디렉토리 절대 경로를 구합니다.
# __file__: 현재 실행 중인 파일의 경로를 담고 있는 내장 변수
path = os.path.dirname(__file__)

# 현재 스크립트 파일이 있는 경로와 CSV 파일명을 합쳐서 전체 경로를 만듭니다.
load_file = os.path.join(path, 'titanic_train.csv')

# pandas의 read_csv 함수를 이용해 CSV 파일을 DataFrame으로 불러옵니다.
titanic_df = pd.read_csv(load_file)

# 데이터가 잘 불러와졌는지 확인하기 위해 상위 3개 행만 출력합니다.
print(titanic_df.head(3))

# 데이터의 전반적인 정보를 확인합니다.
# 전체 행의 개수, 컬럼별 데이터 타입, Non-Null 데이터 개수 등을 알 수 있습니다.
print(f'\n ### train 데이터 정보 ### \n')
print(titanic_df.info())

# --- 결측치(NaN) 처리 시작 ---
# 머신러닝 알고리즘은 NaN 값을 처리하지 못하는 경우가 많으므로 적절한 값으로 채워줘야 합니다.

# Age 컬럼의 NaN 값을 해당 컬럼의 평균값(mean)으로 채웁니다.
# inplace=True: 원본 DataFrame을 직접 수정합니다.
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

# Cabin(선실) 컬럼의 NaN 값을 'N'이라는 문자로 채웁니다.
titanic_df['Cabin'].fillna('N', inplace=True)

# Embarked(정박 항구) 컬럼의 NaN 값을 'N'이라는 문자로 채웁니다.
titanic_df['Embarked'].fillna('N', inplace=True)

# 모든 결측치가 제거되었는지 확인하기 위해 전체 DataFrame의 null 값 개수를 합산하여 출력합니다.
# sum()을 두 번 호출하는 이유: 첫 번째 sum()은 컬럼별 합계, 두 번째 sum()은 전체 합계입니다.
print(f'데이터 세트 NULL 값 개수: {titanic_df.isnull().sum().sum()}')

# --- 데이터 분포 확인 ---
# 주요 범주형 변수들의 값 분포를 확인합니다. 어떤 클래스가 많은지 파악할 수 있습니다.
print(f'Sex 값 분포: \n{titanic_df["Sex"].value_counts()}')
print(f'\n Cabin 값 분포: \n{titanic_df["Cabin"].value_counts()}')
print(f'\n Embarked 값 분포: \n{titanic_df["Embarked"].value_counts()}')

# Cabin(선실) 데이터가 'C85', 'C123' 처럼 되어 있는데, 선실 등급을 나타내는 첫 글자가 중요해 보이므로
# 문자열 슬라이싱(.str[:1])을 통해 첫 글자만 남기고 저장합니다.
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df["Cabin"].head(3))

# --- 데이터 시각화 및 분석 ---

# 성별(Sex)과 생존 여부(Survived)로 그룹화하여 각 그룹의 생존자 수를 카운트합니다.
# 0은 사망, 1은 생존을 의미합니다.
print(titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())

# 성별에 따른 생존 확률을 막대 그래프로 시각화합니다.
# 여성이 남성보다 생존율이 높음을 확인할 수 있습니다.
sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.show() # 그래프 출력

# 객실 등급(Pclass)별 생존 확률을 성별(Sex)로 구분(hue)하여 시각화합니다.
# 부유한(1등급) 사람들의 생존율이 높은지, 성별에 따라 차이가 있는지 확인합니다.
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
plt.show()

# 나이(Age)를 구간별로 나누기 위한 함수를 정의합니다.
# 수치형 데이터인 나이를 범주형 카테고리로 변환하는 작업입니다.
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'   # 나이가 -1 이하인 경우 (데이터 오류 등)
    elif age <= 5: cat = 'Baby'     # 5세 이하 아기
    elif age <= 12: cat = 'Child'   # 12세 이하 어린이
    elif age <= 18: cat = 'Teenager'# 18세 이하 청소년
    elif age <= 25: cat = 'Student' # 25세 이하 학생/청년
    elif age <= 35: cat = 'Young Adult' # 35세 이하 젊은 성인
    elif age <= 60: cat = 'Adult'   # 60세 이하 성인
    else : cat = 'Elderly'          # 60세 초과 노년층

    return cat

# 그래프의 크기를 가로 10인치, 세로 6인치로 설정합니다.
plt.figure(figsize=(10, 6))

# x축에 표시될 카테고리의 순서를 지정합니다. (나이순 정렬을 위해 필요)
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# apply 함수와 lambda 식을 사용하여 'Age' 컬럼의 모든 값에 get_category 함수를 적용합니다.
# 변환된 결과는 'Age_cat'이라는 새로운 컬럼에 저장됩니다.
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))

# 나이 카테고리(Age_cat)별 생존율을 성별(Sex)로 나누어 시각화합니다.
# order 파라미터로 x축의 순서를 지정합니다.
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
plt.show()

# 시각화가 끝났으므로 임시로 만든 'Age_cat' 컬럼은 삭제합니다.
titanic_df.drop(columns=['Age_cat'], axis=1, inplace=True)


# --- 데이터 전처리 함수 정의 (재사용성을 위해 함수화) ---

# 범주형(문자열) 피처를 숫자형으로 인코딩(Label Encoding)하는 함수
def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked'] # 변환할 컬럼 리스트
    for feature in features:
        le = preprocessing.LabelEncoder() # LabelEncoder 객체 생성
        le = le.fit(dataDF[feature])      # 해당 컬럼의 데이터로 인코딩 규칙 학습 (예: female->0, male->1)
        dataDF[feature] = le.transform(dataDF[feature]) # 규칙에 따라 데이터를 숫자로 변환
    
    return dataDF

# 위에서 정의한 encode_features 함수를 호출하여 실제 데이터에 적용합니다.
titanic_df = encode_features(titanic_df)
# 인코딩이 잘 되었는지 상위 5개 행을 확인합니다.
print(titanic_df.head())


# 결측치(Null)를 처리하는 함수 (앞서 수행한 로직을 함수로 묶음)
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True) # 나이: 평균값 대체
    df['Cabin'].fillna('N', inplace=True)            # 선실: 'N' 대체
    df['Embarked'].fillna('N', inplace=True)         # 항구: 'N' 대체
    df['Fare'].fillna(0, inplace=True)               # 요금: 0으로 대체 (혹시 모를 결측치 대비)

    return df

# 모델 학습에 불필요한 피처를 제거하는 함수
def drop_features(dataDF):
    # PassengerId: 단순 식별자
    # Name: 이름 (복잡한 텍스트 분석 없이는 사용 어려움)
    # Ticket: 티켓 번호 (패턴 불명확)
    dataDF.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return dataDF

# 데이터 포맷을 정리하는 함수 (Cabin 첫글자 추출 및 레이블 인코딩)
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 위의 모든 전처리 단계를 순차적으로 실행하는 마스터 함수
def transform_features(df):
    df = fillna(df)          # 1. 결측치 처리
    df = drop_features(df)   # 2. 불필요 피처 제거
    df = format_features(df) # 3. 포맷팅 및 인코딩

    return df

# --- 머신러닝 학습 및 평가 시작 ---

# 데이터를 처음부터 다시 로드합니다. (깨끗한 상태에서 전처리 과정을 일괄 적용하기 위함)
titanic_df = pd.read_csv(load_file)

# 레이블(Target, 정답) 데이터 분리: 'Survived' 컬럼
y_titanic_df = titanic_df['Survived']
# 피처(Input, 학습) 데이터 분리: 'Survived'를 제외한 나머지
X_titanic_df = titanic_df.drop('Survived', axis=1)

# 정의해둔 전처리 함수를 통해 학습 데이터(X)를 변환합니다.
X_titanic_df = transform_features(X_titanic_df)

# 전체 데이터를 학습용(Training)과 테스트용(Test)으로 분리합니다.
# test_size=0.2: 전체의 20%를 테스트 데이터로 사용
# random_state=11: 매번 같은 결과를 얻기 위해 난수 발생 시드 고정
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)


# 사용할 3가지 분류(Classification) 모델 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11) # 결정 트리
rf_clf = RandomForestClassifier(random_state=11) # 랜덤 포레스트
lr_clf = LogisticRegression()                    # 로지스틱 회귀


# 1. DecisionTreeClassifier 학습 및 평가
dt_clf.fit(X_train, y_train) # 학습 데이터로 모델 훈련
dt_pred = dt_clf.predict(X_test) # 테스트 데이터에 대한 예측 수행
# 실제 정답(y_test)과 예측값(dt_pred)을 비교하여 정확도 출력
print(f'DecisionTreeClassifier 정확도: {accuracy_score(y_test, dt_pred):.4f}')


# 2. RandomForestClassifier 학습 및 평가
rf_clf.fit(X_train, y_train) # 학습
rf_pred = rf_clf.predict(X_test) # 예측
print(f'RandomForestClassifier 정확도: {accuracy_score(y_test, rf_pred):.4f}')


# 3. LogisticRegression 학습 및 평가
lr_clf.fit(X_train, y_train) # 학습
lr_pred = lr_clf.predict(X_test) # 예측
print(f'LogisticRegression 정확도: {accuracy_score(y_test, lr_pred):.4f}')


# --- 교차 검증 (Cross Validation) ---

# K-Fold 교차 검증을 수행하는 함수
# folds: 데이터를 몇 개의 그룹으로 나눌지 설정
def exec_kfold(clf, folds=5):
    kfold = KFold(n_splits=folds) # KFold 객체 생성
    scores = [] # 각 폴드에서의 정확도를 저장할 리스트

    # kfold.split(X_titanic_df): 전체 데이터를 학습/검증 인덱스로 분할하여 반복
    # iter_count: 반복 횟수 (0부터 시작)
    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        # 인덱스를 사용하여 학습용과 검증용 데이터 추출
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

        # 모델 학습
        clf.fit(X_train, y_train)
        # 예측
        pred = clf.predict(X_test)
        # 정확도 계산
        accuracy = accuracy_score(y_test, pred)
        scores.append(accuracy) # 결과 리스트에 추가
        print(f'교차 검증 {iter_count+1} 정확도: {accuracy:.4f}')

    # 모든 폴드의 정확도 평균 계산
    mean_score = np.mean(scores)
    print(f'평균 정확도: {mean_score:.4f}')

# DecisionTreeClassifier 모델에 대해 5-폴드 교차 검증 실행
exec_kfold(dt_clf, folds=5)


# --- cross_val_score를 이용한 간편한 교차 검증 ---

# cross_val_score: 위에서 수동으로 작성한 반복문을 내부적으로 처리해주는 함수
# dt_clf: 모델, X_titanic_df: 전체 피처, y_titanic_df: 전체 레이블
# cv=5: 5-폴드 교차 검증
scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)

# 각 폴드별 정확도 출력
for iter_count, accuracy in enumerate(scores):
    print(f'교차 검증 {iter_count+1} 정확도: {accuracy:.4f}')

# 평균 정확도 출력
print(f'평균 정확도: {np.mean(scores):.4f}')


# --- GridSearchCV를 이용한 하이퍼 파라미터 튜닝 ---
# 모델의 성능을 최적화하기 위해 여러 가지 파라미터 조합을 시도해보는 과정

# 튜닝할 파라미터 목록을 딕셔너리로 정의
parameters = {
    'max_depth': [2, 3, 5, 10],       # 트리의 최대 깊이
    'min_samples_split': [2, 3, 5],   # 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf': [1, 5, 8]     # 리프 노드가 되기 위한 최소 샘플 수
}

# GridSearchCV 객체 생성
# estimator: 튜닝할 모델
# param_grid: 시도해볼 파라미터들
# scoring: 평가 기준 (여기서는 정확도)
# cv: 교차 검증 폴드 수
grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)

# 그리드 서치 수행 (학습 데이터를 이용해 최적의 파라미터 탐색 및 교차 검증)
grid_dclf.fit(X_train, y_train)

# 찾아낸 최적의 파라미터 조합 출력
print(f'GridSearchCV 최적 하이퍼 파라미터: {grid_dclf.best_params_}')
# 최적의 파라미터일 때의 최고 정확도 출력
print(f'GridSearchCV 최고 정확도: {grid_dclf.best_score_:.4f}')

# grid_dclf.best_estimator_: 최적의 파라미터로 이미 재학습(refit)된 모델을 반환
best_dclf = grid_dclf.best_estimator_

# 최적의 모델(best_dclf)을 사용하여 테스트 데이터에 대한 예측 및 평가 수행
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print(f'테스트 세트에서의 DecisionTreeClassifier 정확도: {accuracy:.4f}')
