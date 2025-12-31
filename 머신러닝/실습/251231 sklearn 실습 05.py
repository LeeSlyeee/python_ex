from sklearn import preprocessing # 데이터 전처리를 위한 모듈 (LabelEncoder 등 포함)
from sklearn.preprocessing import LabelEncoder # 문자를 숫자로 변환하는 인코더
from sklearn.model_selection import train_test_split # 학습 데이터와 테스트 데이터를 분리하는 함수
from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 분류 모델
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트 분류 모델 (앙상블 기법)
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 분류 모델
from sklearn.metrics import accuracy_score # 모델의 정확도를 계산하는 함수
from sklearn.model_selection import KFold # K-Fold 교차 검증을 위한 클래스
from sklearn.model_selection import cross_val_score # 교차 검증 점수를 계산하는 함수
from sklearn.model_selection import GridSearchCV # 하이퍼 파라미터 튜닝을 위한 그리드 서치

import numpy as np # 수치 계산을 위한 라이브러리
import pandas as pd # 데이터 처리를 위한 라이브러리 (DataFrame 등)
import matplotlib.pyplot as plt # 시각화를 위한 라이브러리
import seaborn as sns # matplotlib 기반의 고급 시각화 라이브러리
import os # 운영체제(파일 경로 등)와 상호작용하기 위한 라이브러리

path = os.path.dirname(__file__)
load_file = os.path.join(path, 'titanic_train.csv')

# csv 파일을 pandas DataFrame으로 로드합니다.
titanic_df = pd.read_csv(load_file)

# 데이터의 상위 3개 행을 출력하여 구조를 확인합니다.
print(titanic_df.head(3))

# 데이터의 정보를 출력합니다. (행/열의 수, 컬럼별 데이터 타입, Null 값 개수 등)
print(f'\n ### train 데이터 정보 ### \n')
print(titanic_df.info())

# 결측치(Null 값) 처리
# Age(나이)의 결측치는 평균값으로 채웁니다.
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
# Cabin(선실 번호)의 결측치는 'N'으로 채웁니다.
titanic_df['Cabin'].fillna('N', inplace=True)
# Embarked(탑승 항구)의 결측치는 'N'으로 채웁니다.
titanic_df['Embarked'].fillna('N', inplace=True)

# 결측치 처리가 잘 되었는지 확인하기 위해 전체 Null 값의 합계를 출력합니다.
print(f'데이터 세트 NULL 값 개수: {titanic_df.isnull().sum().sum()}')


# 범주형 데이터의 값 분포 확인
print(f'Sex 값 분포: \n{titanic_df["Sex"].value_counts()}')
print(f'\n Cabin 값 분포: \n{titanic_df["Cabin"].value_counts()}')
print(f'\n Embarked 값 분포: \n{titanic_df["Embarked"].value_counts()}')


# Cabin 속성의 경우 앞문자만 추출하여 정리합니다. (선실 등급 등 파악 용이)
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df["Cabin"].head(3))


# 성별(Sex)에 따른 생존자(Survived) 수 비교
# counts()를 통해 각 그룹별 데이터 수를 확인합니다.
titanic_df.groupby(['Sex', 'Survived'])['Survived'].count()
# 성별에 따른 생존율을 막대 그래프로 시각화합니다.
sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.show()


# 객실 등급(Pclass)별 성별에 따른 생존율 시각화
# hue를 사용하여 성별로 색을 다르게 표시합니다.
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
plt.show()


# 나이(Age)에 따라 카테고리를 분류하는 함수 정의
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown' # -1 이하인 경우 Unknown (이상치 처리)
    elif age <= 5: cat = 'Baby' # 5세 이하
    elif age <= 12: cat = 'Child' # 12세 이하
    elif age <= 18: cat = 'Teenager' # 18세 이하
    elif age <= 25: cat = 'Student' # 25세 이하
    elif age <= 35: cat = 'Young Adult' # 35세 이하
    elif age <= 60: cat = 'Adult' # 60세 이하
    else : cat = 'Elderly' # 그 외 (61세 이상)

    return cat



# 그래프 크기 설정 (가로 10, 세로 6)
plt.figure(figsize=(10, 6))

# x축의 정렬 순서를 정의합니다.
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# lambda 함수를 사용하여 'Age' 컬럼의 각 값에 get_category 함수를 적용하고, 결과를 새로운 컬럼 'Age_cat'에 저장합니다.
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))

# 나이 카테고리별 성별 생존율을 시각화합니다. order 인자로 x축 순서를 지정합니다.
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
plt.show()

# 시각화 후 불필요해진 'Age_cat' 컬럼을 삭제합니다.
titanic_df.drop(columns=['Age_cat'], axis=1, inplace=True)





# 문자열 카테고리 데이터를 숫자형으로 인코딩하는 함수 정의
def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder() # 라벨 인코더 객체 생성
        le = le.fit(dataDF[feature]) # 데이터에 맞춰 인코딩 학습
        dataDF[feature] = le.transform(dataDF[feature]) # 데이터를 숫자로 변환
    
    return dataDF

# 인코딩 함수를 적용하여 문자를 숫자로 변환합니다.
titanic_df = encode_features(titanic_df)
print(titanic_df.head())
    


# 결측치를 처리하는 함수 (위에서 했던 작업들을 함수로 정리)
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)

    return df

# 불필요한 피처(속성)를 제거하는 함수
def drop_features(dataDF):
    # PassengerId, Name, Ticket은 생존 예측에 큰 영향이 없거나 단순 식별자이므로 제거
    dataDF.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return dataDF

# 레이블 인코딩을 수행하는 함수
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1] # Cabin의 첫 글자만 추출
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder() # LabelEncoder 객체 생성
        le = le.fit(df[feature]) # 학습
        df[feature] = le.transform(df[feature]) # 변환
    return df


# 위에서 정의한 전처리 함수들을 순차적으로 호출하는 함수
def transform_features(df):
    df = fillna(df) # 결측치 처리
    df = drop_features(df) # 불필요 피처 제거
    df = format_features(df) # 레이블 인코딩

    return df


# 원본 데이터를 다시 로드합니다. (전처리 과정을 처음부터 다시 수행하기 위함)
titanic_df = pd.read_csv(load_file)

# 레이블(정답) 데이터 세트 추출
y_titanic_df = titanic_df['Survived']
# 피처(입력) 데이터 세트 추출 (정답 컬럼 제거)
X_titanic_df = titanic_df.drop('Survived', axis=1)

# 피처 데이터 전처리 수행 (결측치 처리, 불필요 컬럼 삭제, 인코딩)
X_titanic_df = transform_features(X_titanic_df)

# 학습 데이터와 테스트 데이터 분리 (테스트 데이터 20%, 랜덤 시드 11)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)


# 3가지 분류 모델 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11) # 의사결정 트리 (random_state로 난수 고정)
rf_clf = RandomForestClassifier(random_state=11) # 랜덤 포레스트
lr_clf = LogisticRegression() # 로지스틱 회귀


# DecisionTreeClassifier 학습 및 예측, 정확도 평가
dt_clf.fit(X_train, y_train) # 학습 수행
dt_pred = dt_clf.predict(X_test) # 테스트 데이터로 예측
print(f'DecisionTreeClassifier 정확도: {accuracy_score(y_test, dt_pred):.4f}')


# RandomForestClassifier 학습 및 예측, 정확도 평가
rf_clf.fit(X_train, y_train) # 학습 수행
rf_pred = rf_clf.predict(X_test) # 테스트 데이터로 예측
print(f'RandomForestClassifier 정확도: {accuracy_score(y_test, rf_pred):.4f}')


# LogisticRegression 학습 및 예측, 정확도 평가
lr_clf.fit(X_train, y_train) # 학습 수행
lr_pred = lr_clf.predict(X_test) # 테스트 데이터로 예측
print(f'LogisticRegression 정확도: {accuracy_score(y_test, lr_pred):.4f}')



# K-Fold 교차 검증을 수행하는 함수 정의
def exec_kfold(clf, folds=5):
    # 폴드 세트를 5개인 KFold 객체를 생성합니다.
    kfold = KFold(n_splits=folds)
    scores = [] # 폴드별 정확도를 저장할 리스트

    # KFold 교차 검증 수행
    # kfold.split()은 입력된 데이터(X_titanic_df)를 학습용/검증용 인덱스로 분할하여 반환합니다.
    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        # 반환된 인덱스를 사용하여 학습용, 검증용 데이터를 추출합니다.
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

        # Classifier 학습
        clf.fit(X_train, y_train)
        # 예측
        pred = clf.predict(X_test)
        # 정확도 계산
        accuracy = accuracy_score(y_test, pred)
        scores.append(accuracy) # 리스트에 정확도 추가
        print(f'교차 검증 {iter_count+1} 정확도: {accuracy:.4f}')

    # 평균 정확도 계산
    mean_score = np.mean(scores)
    print(f'평균 정확도: {mean_score:.4f}')

# DecisionTreeClassifier에 대해 K-Fold 교차 검증 수행 (5개의 폴드)
exec_kfold(dt_clf, folds=5)





# sklearn의 cross_val_score를 이용한 교차 검증 (API를 사용하면 더 간편함)
# cv=5로 설정하여 5-fold 교차 검증 수행, scoring='accuracy'로 정확도 평가
scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)

# 각 폴드별 정확도 출력
for iter_count, accuracy in enumerate(scores):
    print(f'교차 검증 {iter_count+1} 정확도: {accuracy:.4f}')

# 평균 정확도 출력
print(f'평균 정확도: {np.mean(scores):.4f}')


# GridSearchCV를 이용한 하이퍼 파라미터 튜닝
# 테스트할 파라미터들을 딕셔너리 형태로 정의 (트리 깊이, 분할 최소 샘플 수 등)
parameters = {'max_depth': [2, 3, 5, 10], 'min_samples_split': [2, 3, 5], 'min_samples_leaf': [1, 5, 8]}

# GridSearchCV 객체 생성
# estimator=dt_clf, param_grid=parameters, scoring='accuracy', cv=5 (5-fold 교차 검증)
grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
# 학습 데이터로 그리드 서치 수행 (가능한 모든 파라미터 조합에 대해 교차 검증 수행)
grid_dclf.fit(X_train, y_train)

# 최적의 하이퍼 파라미터와 그때의 최고 정확도 출력
print(f'GridSearchCV 최적 하이퍼 파라미터: {grid_dclf.best_params_}')
print(f'GridSearchCV 최고 정확도: {grid_dclf.best_score_:.4f}')

# 최적의 파라미터로 학습된 Estimator를 가져옵니다. (refit=True가 기본값이므로 이미 학습되어 있음)
best_dclf = grid_dclf.best_estimator_


# 최적의 모델로 테스트 데이터 세트에 대한 예측 및 정확도 평가
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print(f'테스트 세트에서의 DecisionTreeClassifier 정확도: {accuracy:.4f}')
