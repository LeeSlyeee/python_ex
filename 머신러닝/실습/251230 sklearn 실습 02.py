from sklearn.datasets import load_iris

# 붓꽃(Iris) 데이터 세트를 로딩합니다.
# load_iris()는 파이썬의 딕셔너리와 유사한 sklearn.utils.Bunch 클래스 객체를 반환합니다.
iris_data = load_iris()

# 반환된 iris_data 객체의 타입을 확인합니다. (<class 'sklearn.utils._bunch.Bunch'>)
print(type(iris_data))

# Bunch 객체가 가지고 있는 키(key) 값들을 출력합니다.
# data, target, target_names, feature_names, DESCR 등이 포함되어 있습니다.
keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들: ', keys)

# 1. feature_names: 피처(특성)들의 이름을 확인합니다.
# 리스트(list) 형태이며, 4개의 피처 이름(sepal length, sepal width, petal length, petal width)을 담고 있습니다.
print('\n feature_name 의 type', type(iris_data['feature_names']))
print('feature_name 의 shape', len(iris_data['feature_names'])) # 리스트의 길이(원소 개수) 출력
print('feature_name', iris_data['feature_names'])

# 2. target_names: 예측해야 할 레이블(클래스)의 실제 이름(문자열)을 확인합니다.
# numpy 배열 형태이며, 3개의 품종 이름(setosa, versicolor, virginica)을 담고 있습니다.
print('\n target_names 의 type', type(iris_data['target_names']))
print('target_names 의 shape', len(iris_data['target_names'])) # 배열의 길이(원소 개수) 출력
print('target_names', iris_data['target_names'])

# 3. data: 실제 데이터(피처값)를 확인합니다.
# numpy 2차원 배열(ndarray) 형태입니다.
# shape은 (150, 4)로, 150개의 샘플(행)과 4개의 피처(열)로 구성되어 있습니다.
print('\n data 의 type', type(iris_data['data']))
print('data 의 shape', iris_data['data'].shape)
print('data', iris_data['data'])

# 4. target: 각 데이터의 정답(레이블) 값을 확인합니다.
# numpy 1차원 배열(ndarray) 형태입니다.
# shape은 (150,)로, 150개의 샘플에 대한 정답(0, 1, 2)이 들어있습니다.
print('\n target 의 type', type(iris_data['target']))
print('target 의 shape', iris_data['target'].shape)
print('target', iris_data['target'])
