
# 필요한 라이브러리 임포트
import sklearn 
import numpy as np # 수치 계산을 위한 라이브러리 (배열, 행렬 연산 등)
import pandas as pd # 데이터 처리를 위한 라이브러리 (DataFrame 등)
import os # 시스템 운영체제 관련 기능을 사용하기 위한 라이브러리 (파일 경로 등)
import matplotlib.pyplot as plt # 데이터 시각화를 위한 라이브러리
import matplotlib.ticker as ticker # 데이터 시각화를 위한 라이브러리


# 데이터 전처리 관련 라이브러리
from sklearn.preprocessing import LabelEncoder # 범주형 데이터를 수치형으로 변환 (Label Encoding)
from sklearn.preprocessing import Binarizer # 이진 분류를 위한 라이브러리

# Estimator(모델) 생성을 위한 기본 클래스
from sklearn.base import BaseEstimator # 커스텀 분류기를 만들 때 상속받아야 하는 기본 클래스

# 데이터셋 분리 및 평가를 위한 라이브러리
from sklearn.model_selection import train_test_split # 학습 데이터와 테스트 데이터 분리
from sklearn.metrics import accuracy_score # 모델의 정확도 평가
from sklearn.metrics import precision_score # 정밀도 평가
from sklearn.metrics import recall_score # 재현율 평가
from sklearn.metrics import confusion_matrix # 혼동 행렬 생성
from sklearn.metrics import precision_recall_curve # 정밀도-재현율 곡선
from sklearn.metrics import f1_score # F1 score 평가
from sklearn.metrics import roc_curve # ROC 곡선
from sklearn.metrics import roc_auc_score # ROC AUC score 평가


# 분류 모델 생성을 위한 라이브러리
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델

# 예제 데이터셋
from sklearn.datasets import load_digits # 손글씨 숫자(digits) 데이터셋 로드



# BaseEstimator를 상속받아 커스텀 분류기 생성
# 이 분류기는 학습(fit) 과정 없이, 성별(Sex)에 따라 생존 여부를 무조건적으로 예측하는 단순한 로직을 가집니다.
# BaseEstimator를 상속받으면 사이킷런의 다른 기능(예: cross_val_score 등)과 호환성을 가질 수 있습니다.
class MyDummyClassifier(BaseEstimator):
    
    # fit 메서드: 모델을 학습시키는 메서드이지만, 여기서는 아무런 학습도 하지 않으므로 pass로 처리합니다.
    def fit(self, X, y=None):
      pass

    # predict 메서드: 학습된 모델(여기서는 규칙 기반)을 사용하여 예측을 수행합니다.
    # 입력된 데이터 X의 'Sex' 컬럼을 확인하여 단순한 규칙으로 생존 여부를 예측합니다.
    def predict(self, X):
      # 입력 데이터의 크기만큼 0으로 채워진 배열 생성 (초기화)
      pred = np.zeros((X.shape[0], 1))
      
      # 데이터의 각 행(row)을 순회하며 예측 수행
      for i in range(X.shape[0]):
        # 성별이 여성이면(1) 생존(0)으로 예측한다고 되어 있으나... 
        # (원본 코드의 로직이 Sex=1일 때 0, else 1로 되어 있음. 
        # 보통 Titanic 데이터에서 Sex=1은 male, 0은 female로 인코딩되거나 반대일 수 있는데,
        # 여기서는 단순히 규칙 기반 분류기의 예시를 보여주기 위함입니다.)
        if X['Sex'].iloc[i] == 1:
          pred[i] = 0 # 사망 예측
        else:
          pred[i] = 1 # 생존 예측
      return pred

# 결측치(Null 값)를 처리하는 함수
def fillna(df):
  # 'Age' 컬럼의 결측치를 평균값으로 채움. 
  # inplace=True 대신 재할당 방식을 사용하여 Pandas 3.0 Future Warning 방지
  df['Age'] = df['Age'].fillna(df['Age'].mean())
  # 'Cabin' 결측치는 'N'으로 채움
  df['Cabin'] = df['Cabin'].fillna('N')
  # 'Embarked' 결측치는 'N'으로 채움
  df['Embarked'] = df['Embarked'].fillna('N')
  # 'Fare' 결측치는 0으로 채움
  df['Fare'] = df['Fare'].fillna(0)
  return df

# 불필요한 속성(컬럼)을 제거하는 함수
def drop_features(df):
  # PassengerId, Name, Ticket 컬럼은 생존 예측에 큰 영향이 없거나 처리가 복잡하므로 제거
  df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
  return df

# 레이블 인코딩을 수행하는 함수
def format_features(df):
  # Cabin(객실 번호)의 첫 글자만 따서 범주 단순화
  df['Cabin'] = df['Cabin'].str[:1]
  
  # 인코딩할 피처 목록
  features = ['Cabin', 'Sex', 'Embarked']
  for feature in features:
    le = LabelEncoder()
    # 해당 피처의 데이터로 인코더 학습
    le = le.fit(df[feature])
    # 학습된 인코더로 데이터 변환 (문자열 -> 숫자)
    df[feature] = le.transform(df[feature])
  return df

# 위에서 정의한 전처리 함수들을 순차적으로 호출하는 함수
def transform_features(df):
  df = fillna(df)
  df = drop_features(df)
  df = format_features(df)
  return df

# 데이터 파일 경로 설정 및 로딩
path = os.path.dirname(__file__)
load_file = os.path.join(path, 'titanic_train.csv')

# CSV 파일 읽기
titanic_df = pd.read_csv(load_file)

# 레이블(정답) 데이터 분리
y_titanic_df = titanic_df['Survived']
# 피처(학습) 데이터 분리 (정답 컬럼 제거)
X_titanic_df = titanic_df.drop('Survived', axis=1)

# 피처 데이터 전처리 수행
X_titanic_df = transform_features(X_titanic_df)

# 학습 데이터와 테스트 데이터 분리 (테스트 데이터 20%)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=0)

# 1. Dummy Classifier 학습 및 예측 실습
# 위에서 정의한 단순 규칙 기반 분류기 객체 생성
myclf = MyDummyClassifier()

# 학습 수행 (실제로는 아무것도 하지 않음)
myclf.fit(X_train, y_train)

# 예측 수행
mypredictions = myclf.predict(X_test)

# 정확도 출력
print(f'Dummy Classifier의 정확도는: {accuracy_score(y_test, mypredictions):.4f}')


# ----------------------------------------------------
# 2. 불균형한 데이터셋에서의 정확도 지표의 함정 실습
# ----------------------------------------------------

# 모든 예측을 0(False)으로 반환하는 가짜 분류기 정의
class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass
    
    # 입력 데이터의 길이만큼 모두 0(False)으로 채운 배열 반환
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

# 사이킷런 내장 데이터셋인 digits(손글씨 숫자) 데이터 로드 (MNIST와 유사)
digits = load_digits()

# 데이터 확인 (픽셀 값들)
print(digits.data)
print(f'### digits.data.shape : {digits.data.shape}')

# 타겟(레이블) 확인 (0~9 숫자)
print(digits.target)
print(f'### digits.target.shape : {digits.target.shape}')


# **불균형 데이터셋 생성**
# target이 7인 경우만 True(1), 나머지는 False(0)으로 변경
# 즉, "숫자 7입니까?"를 맞추는 이진 분류 문제로 변환
print(digits.target == 7)

# True/False를 1/0 정수로 변환
y = (digits.target == 7).astype(int)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)

# 불균형한 레이블 분포 확인
print(f'레이블 테스트 세트 크기: {y_test.shape}')
print('테스트 세트 레이블 0과 1의 분포도')
# 0(7이 아님)이 압도적으로 많고, 1(7임)은 적음을 확인할 수 있음
print(pd.Series(y_test).value_counts())

# 가짜 분류기(무조건 0 예측)로 학습 및 예측 수행
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train, y_train)
fakepred = fakeclf.predict(X_test)

# 정확도 평가
# 무조건 "7이 아니다(0)"라고 찍어도, 데이터의 90%가 7이 아니기 때문에 
# 정확도가 90%가 나오는 역설적인 상황(Accuracy Paradox)을 보여줍니다.
# 결론: 불균형한 데이터셋에서는 정확도(Accuracy)만으로 모델 성능을 평가하면 안 된다.
print(f'모든 예측을 0으로 하여도 정확도는: {accuracy_score(y_test, fakepred):.3f}')



# ----------------------------------------------------
# 3. 오차 행렬, 정밀도, 재현율 실습
# ----------------------------------------------------

# 오차 행렬(Confusion Matrix) 출력
# TN, FP
# FN, TP 형태로 출력됩니다.
print(confusion_matrix(y_test, fakepred))

# 정밀도(Precision)와 재현율(Recall) 계산
# 정밀도: TP / (TP + FP) -> 예측을 Positive로 한 것 중 실제 Positive 비율
# 재현율: TP / (TP + FN) -> 실제 Positive 중 예측을 Positive로 한 비율
print(f'정밀도: {precision_score(y_test, fakepred)}')
print(f'재현율: {recall_score(y_test, fakepred)}')


# 모델 평가 지표를 한 번에 출력하는 도우미 함수 정의
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    print('오차 행렬')  
    print(confusion)
    print(f'정확도: {accuracy:.4f} 정밀도: {precision:.4f} 재현율: {recall:.4f}')


# 타이타닉 데이터셋을 로드하고 전처리 다시 수행 (위의 실습과 이어짐)
titanic_df = pd.read_csv(load_file)
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

# 로지스틱 회귀(Logistic Regression) 모델 생성 및 학습
lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict(X_test)

# 평가 지표 출력
get_clf_eval(y_test, y_pred)  


# predict_proba: 예측 확률을 반환 (0일 확률, 1일 확률)
pred_proba = lr_clf.predict_proba(X_test)
# predict: 예측 결정값(0 또는 1)을 반환
pred = lr_clf.predict(X_test)

print(f'pred_proba() 결과 Shape : {pred_proba.shape}')
# 첫 3개 샘플의 예측 확률 확인 [0일 확률, 1일 확률]
print(f'pred_proba array에서 앞 3개만 샘플로 추출 :\n{pred_proba[:3]}')

# 예측 확률과 예측 결과값을 병합하여 비교
# 확률이 0.5보다 크면 1, 작으면 0으로 예측하는지 확인 가능
pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1, 1)], axis=1)
print(f'두개의 class 중에서 더 큰 확률을 클래스 값으로 예측 :\n{pred_proba_result[:3]}')


# ----------------------------------------------------
# 4. 임계값(Threshold) 조정에 따른 성능 변화 실습 (Binarizer)
# ----------------------------------------------------
# Binarizer: 지정된 기준값(threshold)보다 크면 1, 작거나 같으면 0으로 변환하는 도구

# 예시 데이터 생성
X = [[1, -1, 2], 
     [2, 0, 0], 
     [0, 1.1, 1.2]]

# threshold=1.5로 설정: 1.5보다 크면 1, 아니면 0
binarizer = Binarizer(threshold=1.5)
print(binarizer.fit_transform(X))

# 임계값을 0.5로 설정 (기본적인 로지스틱 회귀의 결정 기준)
custom_threshold = 0.5

# Positive 클래스(1)에 대한 확률만 추출
pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)

# Binarizer를 사용하여 확률값이 0.5보다 크면 1, 아니면 0으로 변환
binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba_1)

# 성능 평가: 기본 predict() 결과와 동일해야 함
get_clf_eval(y_test, custom_predict)


# 임계값을 0.4로 낮춤 -> Positive 예측 확률이 0.4만 넘어도 Positive(1)로 예측
# 이렇게 하면 Recall(재현율)은 높아지지만 Precision(정밀도)은 떨어질 가능성이 있음
custom_threshold = 0.4
pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba_1)

get_clf_eval(y_test, custom_predict)


# 여러 임계값에 대해 반복적으로 평가를 수행하는 함수
thresholds = [0.4, 0.45, 0.50, 0.55, 0.60]

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        # 전달받은 pred_proba_c1을 사용하여 Binarizer 적용
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print(f'임계값: {custom_threshold}')
        # 앞서 정의한 평가 함수 재사용
        get_clf_eval(y_test, custom_predict)

# 함수 호출하여 임계값별 성능 비교
get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds)


# ----------------------------------------------------
# 5. 정밀도-재현율 커브(Precision-Recall Curve) 시각화
# ----------------------------------------------------

# Positive 클래스(1)의 예측 확률 추출
pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1]

# precision_recall_curve 함수로 임계값 변화에 따른 정밀도, 재현율 값 반환
precision, recall, thresholds = precision_recall_curve(y_test, pred_proba_class1)

print(f'반환된 분류 결정 임곗값 배열의 Shape : {thresholds.shape}')
print(f'반환된 precision 배열의 Shape : {precision.shape}')
print(f'반환된 recall 배열의 Shape : {recall.shape}')

print(f'tresholds 5 sample: {thresholds[:5]}')
print(f'precision 5 sample: {precision[:5]}')
print(f'recall 5 sample: {recall[:5]}')

# 샘플 데이터 15단계 건너뛰며 추출
thr_index = np.arange(0, thresholds.shape[0], 15)

print(f'샘플 추출을 위한 임계값 배열의 index 10개: {thr_index}')
print(f'샘플용 10개의 임곗값 : {np.round(thresholds[thr_index], 2)}')

print(f'샘플 임계값별 정밀도 : {np.round(precision[thr_index], 3)}')
print(f'샘플 임계값별 재현율 : {np.round(recall[thr_index], 3)}')


# 정밀도-재현율 변화를 그래프로 시각화하는 함수
def precision_recall_curve_plot(y_test, pred_proba_c1):
    # 정밀도, 재현율, 임계값 계산
    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    # x축을 임계값(Threshold), y축을 정밀도(점선)와 재현율(실선)로 설정
    plt.plot(thresholds, precision[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recall[0:threshold_boundary], label='recall')
    
    # x축의 단위를 0.1 단위로 설정
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()

# 그래프 그리기
precision_recall_curve_plot(y_test, pred_proba[:, 1])

    
# F1 Score 계산
# 정밀도와 재현율의 조화 평균 (어느 한쪽으로 치우치지 않을 때 높음)
f1 = f1_score(y_test, pred)
print(f'F1 스코어: {f1:.4f}')


# 평가 함수 업데이트: F1 Score까지 출력하도록 수정 (재정의)
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    
    f1 = f1_score(y_test, pred)

    print('오차 행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f} 정밀도: {precision:.4f} 재현율: {recall:.4f} F1: {f1:.4f}')

# 임계값 별로 F1 Score까지 포함된 평가 수행
thresholds = [0.4, 0.45, 0.50, 0.55, 0.60]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds)


# ----------------------------------------------------
# 6. ROC Curve와 AUC Score 실습
# ----------------------------------------------------
    
# Positive 클래스 확률
pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1]
print(f'max predict_proba : {np.max(pred_proba_class1)}')

# ROC 곡선을 위한 FPR, TPR, 임계값 계산
fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)
# 첫 번째 임계값은 보통 max + 1로 설정됨
print(f'thresholds[0]: {thresholds[0]}')

# 샘플 추출
thr_index = np.arange(0, thresholds.shape[0], 5)
print(f'샘플 추출을 위한 임계값 배열의 index 10개: {thr_index}')
print(f'샘플용 10개의 임곗값 : {np.round(thresholds[thr_index], 2)}')

print(f'샘플 임계값별 FPR : {np.round(fprs[thr_index], 3)}')
print(f'샘플 임계값별 TPR : {np.round(tprs[thr_index], 3)}')



# ROC Curve 시각화 함수
def roc_curve_plot(y_test, pred_proba_c1):
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    
    # ROC Curve 그리기
    plt.plot(fprs, tprs, label='ROC Curve')
    # 랜덤 수준의 기준선 (대각선) 그리기
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate') # 1 - Specificity(특이도)
    plt.ylabel('True Positive Rate')  # Recall(재현율)
    plt.legend()
    plt.grid()
    plt.show()  

roc_curve_plot(y_test, pred_proba[:, 1])


# ROC AUC Score 계산 (ROC Curve 아래의 면적)
# 1에 가까울수록 좋은 성능
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, pred_proba)
print(f'ROC AUC 값: {roc_auc:.4f}')


# 평가 함수 최종 업데이트: ROC AUC까지 출력하도록 수정
def get_clf_eval(y_test, pred=None, pred_proba=None):
  confusion = confusion_matrix(y_test, pred)
  accuracy = accuracy_score(y_test, pred)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)
  f1 = f1_score(y_test, pred)
  
  # ROC AUC 계산
  roc_auc = roc_auc_score(y_test, pred_proba)
  
  print('오차 행렬')
  print(confusion)
  print(f'정확도: {accuracy:.4f} 정밀도: {precision:.4f} 재현율: {recall:.4f} F1: {f1:.4f} ROC AUC: {roc_auc:.4f}')

    