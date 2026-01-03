# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석과 머신러닝 모델링에 필요한 다양한 라이브러리들을 불러옵니다.
from sklearn.datasets import load_iris              # 붓꽃(Iris) 데이터셋 로드 함수
from sklearn.datasets import load_breast_cancer     # 유방암 데이터셋 로드 함수
from mpl_toolkits.mplot3d import Axes3D             # 3D 차트 생성을 위한 도구
from sklearn.decomposition import PCA               # 주성분 분석(차원 축소) 클래스
from sklearn.model_selection import train_test_split # 학습/검증 데이터 분리 함수
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델 (분류용)
from sklearn.linear_model import SGDClassifier      # 확률적 경사 하강법 분류 모델
from sklearn.neighbors import KNeighborsClassifier  # K-최근접 이웃 분류 모델
from sklearn.svm import SVC                         # 서포트 벡터 머신 분류 모델 (Support Vector Classification)
from sklearn.tree import DecisionTreeClassifier     # 결정 트리 분류 모델
from sklearn.metrics import confusion_matrix        # 오차 행렬 (모델 성능 평가)
from sklearn.metrics import precision_score, recall_score # 정밀도, 재현율 점수 함수
from sklearn.metrics import f1_score                # F1 Score 점수 함수


# 데이터 처리 및 시각화를 위한 필수 라이브러리
import pandas as pd             # 데이터프레임 처리
import numpy as np              # 수치 연산
import matplotlib.pyplot as plt # 기본 시각화
import seaborn as sns           # 통계적 데이터 시각화
import requests                 # 웹 이미지 다운로드 등 HTTP 요청
from io import BytesIO          # 바이트 스트림 처리를 위함 (이미지 로드용)




# =============================================================================
# [2] 데이터 로드 및 확인 - Iris Dataset
# =============================================================================
# Iris 데이터셋을 로드합니다.
iris = load_iris()
print(iris['DESCR']) # 데이터셋에 대한 설명(Description) 출력

data = iris['data']
print(data[:5]) # 데이터의 상위 5개 행(샘플) 출력 (꽃받침/꽃잎의 길이/너비 정보)


# 특성(Feature) 이름 확인
feature_names = iris['feature_names']
print(feature_names)

# 타겟(Label) 데이터 확인 (0, 1, 2 로 구성됨)
target = iris['target']
print(target[45:55]) # 중간 부분의 타겟 값 확인
print(iris['target_names']) # 타겟의 실제 이름 (setosa, versicolor, virginica)


# 데이터프레임 변환 (가독성 및 처리를 위해)
df_iris = pd.DataFrame(data, columns=feature_names)
print(df_iris.head()) # 상위 5개 행 출력

# 타겟 컬럼 추가
df_iris['target'] = target
print(df_iris.head())
print(df_iris)

# =============================================================================
# [3] 데이터 시각화 (Visualization)
# =============================================================================
# Sepal(꽃받침) 기준 산점도 시각화
sns.scatterplot(x='sepal width (cm)', y='sepal length (cm)', hue='target', palette='muted', data=df_iris)
plt.title('Sepal')
plt.show()


# Petal(꽃잎) 기준 산점도 시각화
sns.scatterplot(x='petal width (cm)', y='petal length (cm)', hue='target', palette='muted', data=df_iris)
plt.title('Petal')
plt.show()


# 3D 시각화 (PCA로 4차원 -> 3차원 축소)
# PCA(Principal Component Analysis): 데이터의 분산을 최대한 보존하면서 차원을 줄이는 기법
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d') # 3D 축 생성
ax.view_init(elev=-150, azim=110) # 뷰 포인트 설정
X_reduced = PCA(n_components=3).fit_transform(df_iris.drop('target', axis=1)) # 4개 특성을 3개로 축소
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=df_iris['target'], # 산점도 그리기
cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Iris 3D")
ax.set_xlabel("x")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("y")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("z")
ax.zaxis.set_ticklabels([])
plt.show()


# =============================================================================
# [4] 데이터 분할 (Train/Validation Split)
# =============================================================================
# 학습 데이터와 검증 데이터로 분할합니다. (기본 75:25 비율)
x_train, x_valid, y_train, y_valid = train_test_split(df_iris.drop('target', axis=1), df_iris['target'])
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)

# 타겟 값의 분포 확인 (랜덤 분할 시 특정 클래스가 쏠릴 수 있음)
sns.countplot(x=y_train)
plt.show()


# stratify 옵션 사용: 타겟 값의 비율을 유지하며 분할
# 분류 문제에서는 stratify를 사용하는 것이 데이터 편향을 막는 데 유리합니다.
x_train, x_valid, y_train, y_ = train_test_split(df_iris.drop('target', axis=1), df_iris['target'], stratify=df_iris['target'])
sns.countplot(x=y_train) # 클래스별 균등하게 분포된 것을 확인 가능
plt.show()

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)


# =============================================================================
# [5] 모델링 1: 로지스틱 회귀 (Logistic Regression)
# =============================================================================
# 로지스틱 회귀: 분류를 위한 선형 모델 (이름은 회귀지만 실제로는 분류에 사용)
# 시그모이드 함수를 사용하여 확률(0~1)을 예측
model = LogisticRegression()
model.fit(x_train, y_train) # 모델 학습

# 예측 및 성능 평가
prediction = model.predict(x_valid)
print(prediction[:5]) # 예측 결과 5개 확인
print((prediction == y_valid).mean()) # 정확도 계산 (맞춘 비율)


# 로지스틱 회귀 개념 설명 이미지 출력
response = requests.get('https://machinelearningnotepad.files.wordpress.com/2018/04/yk1mk.png')
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.show()


# =============================================================================
# [6] 모델링 2: SGD Classifier (확률적 경사 하강법)
# =============================================================================
# SGD (Stochastic Gradient Descent): 데이터를 한 개씩(혹은 미니배치) 보며 가중치를 업데이트하는 방식
# 대용량 데이터 처리에 유리함
sgd = SGDClassifier(random_state=0)
sgd.fit(x_train, y_train)

prediction = sgd.predict(x_valid)
print((prediction == y_valid).mean()) # 정확도 출력

# 하이퍼파라미터 튜닝 예시 (penalty='l1': L1 규제 적용)
# L1 규제는 불필요한 특성의 가중치를 0으로 만들어 특성 선택 효과가 있음
sgd = SGDClassifier(penalty='l1', random_state=0, n_jobs=-1)
print(sgd.fit(x_train, y_train))
prediction = sgd.predict(x_valid)
print((prediction == y_valid).mean())


# KNN(K-Nearest Neighbors) 개념 설명 이미지 출력
response = requests.get('https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final_a1mrv9.png')
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.show()


# =============================================================================
# [7] 모델링 3: K-최근접 이웃 (K-Neighbors Classifier)
# =============================================================================
# KNN: 가장 가까운 K개의 이웃 데이터를 보고 다수결로 분류하는 알고리즘
knc = KNeighborsClassifier() # 기본값 (n_neighbors=5)
print(knc.fit(x_train, y_train))
knc_pred = knc.predict(x_valid)
print((knc_pred == y_valid).mean()) # 정확도 출력

# 이웃의 수(n_neighbors)를 9로 변경하여 테스트
knc = KNeighborsClassifier(n_neighbors=9)
knc.fit(x_train, y_train)
knc_pred = knc.predict(x_valid)
print((knc_pred == y_valid).mean())


# SVC(Support Vector Machine) 개념 설명 이미지 출력
response = requests.get('https://csstudy.files.wordpress.com/2011/03/screen-shot-2011-02-28-at-5-53-26-pm.png')
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.show()


# =============================================================================
# [8] 모델링 4: 서포트 벡터 머신 (SVC)
# =============================================================================
# SVM: 클래스를 구분하는 최적의 경계선(결정 경계)을 찾는 알고리즘
svc = SVC(random_state=0,)
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_valid)

print(svc)
print((svc_pred == y_valid).mean()) # 정확도 출력
print(svc_pred[:5])
print(svc.decision_function(x_valid)[:5]) # 결정 함수 값 (경계로부터의 거리) 확인


# 결정 트리(Decision Tree) 개념 설명 이미지 출력
response = requests.get('https://savioglobal.com/wp-content/uploads/2023/08/image-7.png')
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.show()


# =============================================================================
# [9] 모델링 5: 결정 트리 (Decision Tree Classifier)
# =============================================================================
# Decision Tree: 스무고개처럼 질문을 통해 데이터를 분류해 나가는 구조
dtc = DecisionTreeClassifier(random_state=0)
print(dtc.fit(x_train, y_train))
dtc_pred = dtc.predict(x_valid)
print((dtc_pred == y_valid).mean()) # 정확도 출력


# =============================================================================
# [10] 불균형 데이터 처리 실습 - Breast Cancer Dataset
# =============================================================================
# 유방암 데이터셋 로드
cancer = load_breast_cancer()
print(cancer['DESCR']) # 데이터 설명
data = cancer['data']
target = cancer['target']
feature_names=cancer['feature_names']


# 데이터프레임 생성
df = pd.DataFrame(data=data, columns=feature_names)
df['target'] = cancer['target']
print(df.head())


# 불균형 데이터 상황 연출 (Imbalanced Data Simulation)
# 양성(1) 데이터는 그대로 두고, 음성(0) 데이터는 5개만 가져와서 강제로 불균형하게 만듦
pos = df.loc[df['target']==1]
neg = df.loc[df['target']==0]

print(pos)
print(neg)


# 데이터 합치기
sample = pd.concat([pos, neg[:5]], sort=True)
print(sample)

# 학습/테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(sample.drop('target', axis=1), sample['target'], random_state=42)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print((pred == y_test).mean()) # 정확도 계산

# 무조건 1(양성)로만 예측하는 가짜 예측기 생성
# 불균형 데이터에서는 무조건 다수 클래스로 예측해도 정확도가 높게 나오는 함정이 있음
my_prediction = np.ones(shape=y_test.shape)
print((my_prediction == y_test).mean()) # 정확도만 보면 모델과 비슷하게 높음 (정확도의 역설)


# 오차 행렬 (Confusion Matrix) 확인
# TN(True Neg), FP(False Pos), FN(False Neg), TP(True Pos) 개수 확인
print(confusion_matrix(y_test, pred))

# 히트맵 시각화
sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Reds', )
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()


# 오차 행렬 개념 설명 이미지 출력
response = requests.get('https://dojinkimm.github.io/assets/imgs/ml/handson_3_1.png')
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.axis('off')
plt.show()


# =============================================================================
# [11] 성능 평가 지표 (Evaluation Metrics)
# =============================================================================
# 정밀도(Precision): 양성이라고 예측한 것 중 실제 양성의 비율 (TP / (TP + FP))
# 재현율(Recall): 실제 양성인 것 중 양성으로 예측한 비율 (TP / (TP + FN))
print(precision_score(y_test, pred))
print(recall_score(y_test, pred))

# 단순 계산 비교 (검증용)
print(88/90)

# F1 Score: 정밀도와 재현율의 조화 평균 (불균형 데이터에서 유용한 지표)
print(f1_score(y_test, pred))