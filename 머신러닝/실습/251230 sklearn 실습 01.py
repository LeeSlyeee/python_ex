import sklearn
# 붓꽃(Iris) 데이터 세트를 불러오기 위한 함수입니다.
from sklearn.datasets import load_iris
# 의사결정트리(Decision Tree) 분류 알고리즘을 구현한 클래스입니다.
from sklearn.tree import DecisionTreeClassifier
# 학습 데이터와 테스트 데이터를 쉽게 분리해주는 함수입니다.
from sklearn.model_selection import train_test_split
# 예측 결과의 정확도(Accuracy)를 측정하기 위한 함수입니다.
from sklearn.metrics import accuracy_score
# 데이터를 표 형태(DataFrame)로 변환하고 분석하기 위해 사용합니다.
import pandas as pd



# 붓꽃(iris) 데이터 세트를 로딩합니다. 
# load_iris() 함수는 딕셔너리와 유사한 Bunch 객체를 반환합니다.
iris = load_iris()

# iris.data는 붓꽃의 피처(Feature)만으로 된 데이터를 numpy 가지고 있습니다.
# 피처는 sepal length, sepal width, petal length, petal width 4가지입니다.
iris_data = iris.data

# iris.target은 붓꽃 데이터 세트의 레이블(결정 값) 데이터를 numpy 가지고 있습니다.
# 0: Setosa, 1: Versicolor, 2: Virginica를 의미합니다.
iris_label = iris.target

print('iris target값: ', iris_label)
print('iris target명: ', iris.target_names) # 타겟(레이블)의 이름(문자열)을 확인합니다.

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다.
# columns 파라미터에는 feature_names를 사용하여 각 컬럼의 이름을 지정합니다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

# 정답(Label) 데이터를 'label'이라는 새로운 컬럼으로 추가합니다.
iris_df['label'] = iris_label

# 생성된 DataFrame의 앞쪽 3개 행을 출력하여 데이터 구조를 확인합니다.
print(iris_df.head(3))



# 학습 데이터와 테스트 데이터를 분리합니다.
# test_size=0.2: 전체 데이터의 20%를 테스트 데이터로 사용, 나머지 80%는 학습 데이터로 사용합니다.
# random_state=11: 실습용 난수 발생 시드값입니다. 같은 값을 지정하면 여러 번 수행해도 동일한 결과가 나옵니다.
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

# 의사결정트리(DecisionTreeClassifier) 객체를 생성합니다.
# random_state=11: 코드를 재실행해도 동일한 결과가 나오도록 시드를 고정합니다.
dt_clf = DecisionTreeClassifier(random_state=11)

# 생성된 dt_clf 객체에 학습용 데이터(X_train)와 학습용 레이블(y_train)을 입력하여 학습(Training)을 수행합니다.
dt_clf.fit(X_train, y_train)

# 학습이 완료된 dt_clf 객체를 이용하여 테스트 데이터(X_test)에 대한 예측을 수행합니다.
# 예측 결과는 결정된 클래스 값(0, 1, 2)들의 리스트로 반환됩니다.
y_pred = dt_clf.predict(X_test)

pred = dt_clf.predict(X_test)
print('예측값: ', pred)

# accuracy_score(실제값, 예측값) 함수를 사용하여 예측 정확도를 계산합니다.
# f-string을 사용하여 예측 정확도를 소수점 4자리까지 출력합니다.
print(f'예측 정확도:{accuracy_score(y_test, y_pred):.4f}')
