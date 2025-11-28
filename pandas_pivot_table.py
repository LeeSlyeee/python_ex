import pandas as pd
import numpy as np
import seaborn as sns


# # pivot_table(data,values=None,index=None,columns=None,aggfunc='mean',margins=False,margins_name='All')
# #     data : 분석할 데이터 프레임. 메서드 형식일때는 필요하지 않음 ex)df1.pivot_table()
# #     values : 분석할 데이터 프레임에서 분석할 열
# #     index : 행 인덱스로 들어갈 키열 또는 키열의 리스트
# #     columns : 열 인덱스로 들어갈 키열 또는 키열의 리스트
# #     fill_value : NaN이 표출될 때 대체값 지정
# #     margins : 모든 데이터를 분석한 결과를 행으로 표출할 지 여부
# #     margins_name : margins가 표출될 때 그 열(행)의 이름`

# # 피봇테이블을 작성할 때 반드시 설정해야 되는 인수
# #     data : 사용 데이터 프레임
# #     index : 행 인덱스로 사용할 필드(기준 필드로 작용됨)
# #     인덱스 명을 제외한 나머지 값(data)은 수치 data 만 사용함
# #     기본 함수가 평균(mean)함수 이기 때문에 각 데이터의 평균값이 반환



# data = { # 딕셔너리 형태로 데이터프레임에 들어갈 데이터를 정의합니다.
#     "도시": ["서울", "서울", "서울", "부산", "부산", "부산", "인천", "인천"],
#     "연도": ["2015", "2010", "2005", "2015", "2010", "2005", "2015", "2010"],
#     "인구": [9904312, 9631482, 9762546, 3448737, 3393191, 3512547, 2890451, 263203],
#     "지역": ["수도권", "수도권", "수도권", "경상권", "경상권", "경상권", "수도권", "수도권"]
# }
# columns = ["도시", "연도", "인구", "지역"] # 데이터프레임의 열 순서를 정의합니다.

# df1 = pd.DataFrame(data, columns=columns) # 정의된 'data'와 'columns'를 사용하여 데이터프레임 'df1'을 생성합니다.
# print(df1) # 생성된 데이터프레임 'df1'을 출력합니다.
# # 결과: 도시, 연도, 인구, 지역 열을 가진 8행의 데이터프레임이 출력됩니다.


# # # 첫 번째 피벗 테이블 (단일 인덱스)
# # # 각 도시에 대한 연도별 인구
# print(df1.pivot_table(index="도시", columns='연도',values='인구'))
# # pivot_table 함수를 사용하여 데이터를 재구조화합니다.
# # index="도시": '도시' 열의 고유값을 "새로운 행 인덱스"로 사용합니다. (서울, 부산, 인천)
# # columns='연도': '연도' 열의 고유값을 "새로운 열 이름"으로 사용합니다. (2005, 2010, 2015)
# # values='인구': 새로운 테이블을 채울 "데이터 값"으로 '인구' 열의 값을 사용합니다.
# # aggfunc(집계 함수)를 지정하지 않았으므로, 기본값인 'mean' (평균)이 적용되지만,
# # 이 데이터의 경우 index-columns 조합당 하나의 값만 존재하므로 원래 값이 그대로 표시됩니다.



# # # 두 번째 피벗 테이블 (다중 인덱스)
# # # 각 지역별 도시에 대한 연도별 인구
# print(df1.pivot_table(index=["지역","도시"],columns="연도", values="인구"))
# # index=["지역","도시"]: '지역'과 '도시' 열을 조합하여 "다중 행 인덱스(MultiIndex)"로 사용합니다.
# #                         (예: ('경상권', '부산'), ('수도권', '서울'), ('수도권', '인천'))
# # columns="연도": '연도' 열의 고유값을 "열 이름"으로 사용합니다. (2005, 2010, 2015)
# # values="인구": 새로운 테이블을 채울 "데이터 값"으로 '인구' 열의 값을 사용합니다.
# # 이 테이블은 '지역' 내 '도시' 별 '연도' 인구 데이터를 보여주며,
# # 인덱스가 계층적으로 구성되어 데이터의 그룹화된 보기가 가능합니다.








# # # 1. 데이터 불러오기 및 데이터프레임 생성
# # Seaborn에 내장된 'titanic' 데이터셋을 불러오고, 특정 열('age', 'sex', 'class', 'fare', 'survived')만 선택하여 데이터프레임 'df'를 생성합니다.
# df = sns.load_dataset('titanic')[['age','sex','class','fare','survived']] 
# # print(df.head()) # 생성된 데이터프레임의 상위 5개 행을 출력하여 데이터 구조를 확인합니다.
# # 결과: 선택된 5개 열을 가진 타이타닉 데이터의 일부가 출력됩니다.


# # # 2. 피벗 테이블 생성 (평균 나이 계산)
# pdf1 = pd.pivot_table( # pd.pivot_table 함수를 사용하여 데이터를 재구조화하고 집계합니다.
#     df,               # 사용할 데이터프레임 (타이타닉 데이터)
#     index = 'class',  # 'class' 열의 고유값을 "새로운 행 인덱스"로 사용합니다. (First, Second, Third)
#     columns = 'sex',  # 'sex' 열의 고유값을 "새로운 열 이름"으로 사용합니다. (female, male)
#     values='age',     # 새로운 테이블을 채울 "데이터 값"으로 'age' 열을 사용합니다.
#     aggfunc='mean',   # 인덱스와 컬럼의 각 조합(예: First class 여성)에 대해 'age'의 "평균(mean)"을 계산하도록 지정합니다.
#     observed = False  # NaN 값을 포함하는 행과 열은 제외하고 계산합니다. 향후 버전에서는 기본값이 True로 바뀔예정이라 명시적으로 표기함.
# )
# print(pdf1) # 생성된 피벗 테이블 'pdf1'을 출력합니다.
# # 결과: First/Second/Third 클래스별, 여성/남성 승객의 평균 나이가 행렬 형태로 출력됩니다.


# # # 3. 피벗 테이블 (다중 집계) 주석
# pdf1 = pd.pivot_table( # pd.pivot_table 함수를 사용하여 데이터를 재구조화하고 집계합니다.
#     df,               # 사용할 데이터프레임 (타이타닉 데이터)
#     index = 'class',  # 'class' 열의 고유값을 "새로운 행 인덱스"로 사용합니다. (First, Second, Third)
#     columns = 'sex',  # 'sex' 열의 고유값을 "새로운 열 이름"으로 사용합니다. (female, male)
#     values='survived',# 새로운 테이블을 채울 "데이터 값"으로 'survived' (생존 여부, 0 또는 1) 열을 사용합니다.
#     aggfunc=['mean','sum'], # "다중 집계 함수"를 리스트 형태로 지정합니다.
#     observed = False  # NaN 값을 포함하는 행과 열은 제외하고 계산합니다. 향후 버전에서는 기본값이 True로 바뀔예정이라 명시적으로 표기함.
#     # 'mean': 각 그룹별 "생존율" (survived=1의 평균)을 계산합니다.
#     # 'sum': 각 그룹별 "생존자 수" (survived=1의 합계)를 계산합니다.
# )
# print(pdf1) # 생성된 피벗 테이블 'pdf1'을 출력합니다.
# # 결과: 행에는 'class', 열에는 'mean'과 'sum' 집계 함수 아래 'sex'가 중첩된 "다중 열 인덱스(MultiIndex)"를 가진 테이블이 출력됩니다.


# # # 4. 피벗 테이블 생성 (다중 인덱스, 다중 값, 다중 집계)
# pdf3 = pd.pivot_table( # pd.pivot_table 함수를 사용하여 데이터를 재구조화하고 집계합니다.
#     df,               # 사용할 데이터프레임 (타이타닉 데이터)
#     index = ['class','sex'], # 'class'와 'sex' 열을 조합하여 "다중 행 인덱스(MultiIndex)"로 사용합니다.
#                            # (예: First 클래스의 female, First 클래스의 male, ...)
#     columns = 'survived',    # 'survived' (생존 여부: 0=사망, 1=생존) 열의 고유값을 "새로운 열 이름"으로 사용합니다. (0, 1)
#     values = ['age','fare'], # 새로운 테이블을 채울 "데이터 값"으로 'age'와 'fare' (운임) 열 "두 개"를 사용합니다.
#     aggfunc = ['mean','max'],# "다중 집계 함수"를 리스트 형태로 지정합니다.
#                            # 'mean': 각 그룹별 나이와 운임의 "평균"을 계산합니다.
#                            # 'max': 각 그룹별 나이와 운임의 "최댓값"을 계산합니다.
#     observed = False       # NaN 값을 포함하는 행과 열은 제외하고 계산합니다.
# )
# print(pdf3) # 생성된 피벗 테이블 'pdf3'을 출력합니다.
# # 결과:
# # 1. 행 인덱스는 'class'와 'sex'로 구성된 2단계 계층 구조를 갖습니다.
# # 2. 열 인덱스는 'mean'/'max' (집계 함수), 'age'/'fare' (값), '0'/'1' (생존 여부)로 구성된 "3단계 계층 구조(MultiIndex)"를 갖습니다.
# # 3. 이 테이블은 각 클래스/성별 그룹에서 생존한 사람(1)과 사망한 사람(0)의 평균 나이/운임, 최대 나이/운임을 한눈에 보여줍니다.








# # # 그룹 분석
 
# # 만약 키가 지정하는 조건에 맞는 데이터가 하나 이상이라서 데이터 그룹을 이루는 경우에는 그룹의 특성을 보여주는 그룹분석(group analysis)을 해야 함
# # 그룹분석은 피봇테이블과 달리 키에 의해서 결정되는 데이터가 여러개가 있을 경우 미리 지정한 연산을 통해 그 그룹 데이터의 대표값을 계산 하는 것
# # 판다스에서는 groupby 메서드를 사용하여 아래 내용 처럼 그룹분석을 진행
# # 분석하고자 하는 시리즈나 데이터프레임에 groupby 메서드를 호출하여 그룹화 수행
# # 그룹 객체에 대해 그룹연산을 수행

# # groupby 메서드
# # groupby 메서드는 데이터를 그룹 별로 분류하는 역할을 함
 
# # groupby 메서드의 인수
# # 열 또는 열의 리스트
# # 행 인덱스
# # 연산 결과로 그룹 데이터를 나타내는 GroupBy 클래스 객체를 반환
# # 이 객체에는 그룹별로 연산을 할 수 있는 그룹연산 메서드가 있음

 
# # GroupBy 클래스 객체의 그룹연산 메서드
# # size, count: 그룹 데이터의 갯수
# # mean, median, min, max: 그룹 데이터의 평균, 중앙값, 최소, 최대
# # sum, prod, std, var, quantile : 그룹 데이터의 합계, 곱, 표준편차, 분산, 사분위수
# # first, last: 그룹 데이터 중 가장 첫번째 데이터와 가장 나중 데이터
 
 
# # 이 외에도 많이 사용되는 그룹 연산
# # agg, aggregate
# # 만약 원하는 그룹연산이 없는 경우 함수를 만들고 이 함수를 agg에 전달한다.
# # 또는 여러가지 그룹연산을 동시에 하고 싶은 경우 함수 이름 문자열의 리스트를 전달한다.
 
# # describe
# # 하나의 그룹 대표값이 아니라 여러개의 값을 데이터프레임으로 구한다.
 
# # apply
# # describe 처럼 하나의 대표값이 아닌 데이터프레임을 출력하지만 원하는 그룹연산이 없는 경우에 사용한다.
 
# # transform
# # 그룹에 대한 대표값을 만드는 것이 아니라 그룹별 계산을 통해 데이터 자체를 변형한다.



# np.random.seed(0) # 난수 생성 시드(seed)를 0으로 설정하여 매번 동일한 결과를 얻도록 합니다. (여기서는 직접적인 난수 생성은 없지만 관행적으로 사용될 수 있음)
# df2 = pd.DataFrame({ # 딕셔너리 형태로 데이터프레임 'df2'를 생성합니다.
#     'key1': ['A', 'A', 'B', 'B', 'A'], # 그룹화할 첫 번째 키
#     'key2': ['one', 'two', 'one', 'two', 'one'], # 두 번째 키
#     'data1': [1, 2, 3, 4, 5], # 값 데이터 1
#     'data2': [10, 20, 30, 40, 50] # 값 데이터 2
# })
# print(df2) # 생성된 데이터프레임 'df2'를 출력합니다.


# # --- groupby() 객체 생성 및 출력 ---
# # 'key1' 열의 고유값('A', 'B')을 기준으로 데이터프레임을 그룹화하여 'groups' 객체를 생성합니다.
# groups = df2.groupby(df2.key1) 
# print(groups) 
# # 결과: <pandas.core.groupby.generic.DataFrameGroupBy object at 0x...> 와 같은 형태로 출력됩니다.
# # 이는 그룹화된 결과가 메모리상에 'DataFrameGroupBy' 객체 형태로 존재하며,
# # 이 객체에 대해 집계 함수(sum, mean, count 등)를 적용해야 실제 그룹별 결과가 계산됨을 의미합니다.


# # --- 그룹 정보 접근 및 출력 ---
# print(groups.groups)
# # groups.groups: 그룹화된 결과의 상세 정보를 딕셔너리 형태로 출력합니다.
# # 키(Key)는 그룹 이름('A', 'B')이고, 값(Value)은 해당 그룹에 속하는 "원래 데이터프레임의 행 인덱스" 리스트입니다.
# # 예: {'A': [0, 1, 4], 'B': [2, 3]}

# print(groups.groups.keys())
# # groups.groups.keys(): 그룹 이름을 딕셔너리 키 형태로 출력합니다.
# # 예: dict_keys(['A', 'B'])

# print(groups.groups['A'])
# # groups.groups['A']: 'A' 그룹에 속하는 행의 인덱스만 추출하여 출력합니다.
# # 예: [0, 1, 4]


# # --- 그룹화 객체를 데이터프레임으로 변환 및 추출 ---
# print(pd.DataFrame(groups))
# # 결과: GroupBy 객체를 (그룹 이름, 해당 그룹의 데이터프레임) 쌍의 리스트로 간주하여 새로운 데이터프레임으로 변환합니다.
# # 열 0: 그룹 이름 ('A', 'B')
# # 열 1: 해당 그룹에 속하는 데이터프레임 객체

# print(pd.DataFrame(groups).loc[0].values)
# # 변환된 데이터프레임의 첫 번째 행(인덱스 0)을 선택하고(그룹 'A'), 그 값(values)을 출력합니다.
# # 결과: ['A' df_for_A] (그룹 이름 'A'와 'A' 그룹에 해당하는 데이터프레임 객체)

# print(pd.DataFrame(groups).loc[1].values)
# # 변환된 데이터프레임의 두 번째 행(인덱스 1)을 선택하고(그룹 'B'), 그 값(values)을 출력합니다.
# # 결과: ['B' df_for_B] (그룹 이름 'B'와 'B' 그룹에 해당하는 데이터프레임 객체)

# print(pd.DataFrame(groups).loc[1].values[1])
# # 변환된 데이터프레임의 두 번째 행(인덱스 1, 그룹 'B')의 두 번째 값(인덱스 1, 데이터프레임 객체)을 출력합니다.
# # 결과: 'B' 그룹에 해당하는 "실제 데이터프레임"이 출력됩니다.


# # --- 집계 연산 수행 ---
# print(groups.sum())
# # 그룹별로 "모든 숫자형 열"('data1', 'data2')에 대해 합계(sum)를 계산합니다.
# # 결과: 'key1'('A', 'B')을 인덱스로, 'data1'과 'data2'의 합계를 값으로 하는 데이터프레임이 출력됩니다.

# print(groups['data1'].sum())
# # 'groups' 객체에서 'data1' 열만 선택하여 그룹별 합계를 계산합니다. (Series 반환)
# # 결과: 'key1'별 'data1'의 합계가 Series 형태로 출력됩니다.

# print(groups.sum()['data1'])
# # 위와 동일한 결과를 얻는 다른 방법: 전체 그룹 합계를 구한 후 'data1' 열을 선택합니다. (Series 반환)
# # 결과: 'key1'별 'data1'의 합계가 Series 형태로 출력됩니다.

# print(groups[['data1']].sum())
# # 'groups' 객체에서 'data1' 열을 리스트로 감싸서 선택하여 그룹별 합계를 계산합니다. (DataFrame 반환)
# # 결과: 'key1'별 'data1'의 합계가 "DataFrame" 형태로 출력됩니다.

# print(groups[['data2']].sum())
# # 'groups' 객체에서 'data2' 열을 리스트로 감싸서 선택하여 그룹별 합계를 계산합니다. (DataFrame 반환)
# # 결과: 'key1'별 'data2'의 합계가 "DataFrame" 형태로 출력됩니다.








# # # Iris 데이터셋 GroupBy 연산

# # 1. 데이터 불러오기 및 그룹화
# iris = sns.load_dataset("iris") # Seaborn에 내장된 'iris' 데이터셋을 불러와 데이터프레임 'iris'를 생성합니다.
# # 이 데이터셋은 150개의 꽃(행)에 대한 꽃받침(sepal)과 꽃잎(petal)의 길이/너비 측정값과 종(species) 정보를 담고 있습니다.

# print(iris) # 데이터프레임 'iris' 전체를 출력합니다.

# i_groups = iris.groupby(iris.species) # 'iris' 데이터프레임을 'species' (붓꽃 종: setosa, versicolor, virginica) 열을 기준으로 그룹화하고 GroupBy 객체 'i_groups'를 생성합니다.



# # 2. 기본 집계 연산
# print(i_groups.mean()) # 그룹별(종별)로 "모든 숫자형 열"('sepal_length', 'sepal_width', 'petal_length', 'petal_width')의 "평균(mean)"을 계산합니다.
# # 결과: 3행 4열의 데이터프레임이 출력됩니다.

# print(pd.DataFrame(i_groups).loc[1].values[1]['sepal_width'].max())
# # 그룹화 객체를 데이터프레임으로 변환합니다. (loc[1]은 두 번째 그룹, 즉 'versicolor' 그룹)
# # .values[1]은 해당 그룹의 실제 데이터프레임 객체를 추출합니다.
# # 추출된 versicolor 데이터프레임에서 'sepal_width' 열을 선택하고 "최댓값(max)"을 계산합니다.
# # 결과: versicolor 종의 꽃받침 너비 최댓값이 출력됩니다.

# print(i_groups.petal_length.sum()) # 그룹별(종별)로 'petal_length' 열의 "합계(sum)"를 계산합니다. (Series 반환)

# print(i_groups.max()/i_groups.min())
# # 그룹별로 "최댓값(max)"을 계산한 결과 데이터프레임을 그룹별 "최솟값(min)"을 계산한 결과 데이터프레임으로 나눕니다.
# # 결과: 종별 각 측정값의 "최댓값 대 최솟값 비율"이 계산되어 데이터프레임으로 출력됩니다.



# # 3. 사용자 정의 함수 (UDF) 적용: agg와 apply
# # A. 사용자 정의 함수 정의 및 데이터프레임 적용
# def peak_to_peak_ratio(x):
#     return x.max() / x.min()
# # 'x' (Series 또는 DataFrame 열)의 최댓값을 최솟값으로 나눈 비율을 반환하는 사용자 정의 함수를 정의합니다. (최대-최소 비율)

# peak_to_peak_ratio(iris.sepal_length) # 전체 'sepal_length' 열에 대해 함수를 테스트 실행합니다. (그룹화 없이 전체 데이터에 적용)

# print(i_groups.agg(peak_to_peak_ratio))
# # 그룹별로 "각 열"에 대해 정의된 'peak_to_peak_ratio' 함수를 "집계(aggregate)"합니다.
# # 결과: 그룹(종)별 각 측정값의 (최댓값 / 최솟값) 비율이 데이터프레임으로 출력됩니다. 'agg'는 집계 결과를 반환합니다.

# print(i_groups.apply(peak_to_peak_ratio))
# # 그룹별로 "각 열"에 대해 정의된 'peak_to_peak_ratio' 함수를 "적용(apply)"합니다. (이 경우 agg와 동일한 집계 결과를 반환)
# # 결과: 'agg'와 동일한 결과가 출력되지만, 'apply'는 더 복잡한 그룹별 변형 및 필터링 작업도 처리할 수 있습니다.


# # B. apply를 이용한 데이터 변형 (정렬 후 상위 3개 추출)
# def top3_petal_length(df):
#     # 입력으로 받은 그룹 데이터프레임(df)을 'petal_length' 기준으로 내림차순 정렬하고 상위 3개 행만 반환하는 함수를 정의합니다.
#     return df.sort_values(by="petal_length", ascending=False)[:3]

# top3_petal_length(iris) # 전체 'iris' 데이터프레임에 대해 함수를 테스트 실행합니다. (전체에서 꽃잎 길이 상위 3개 행 반환)

# print(i_groups.apply(top3_petal_length))
# # 그룹별로(종별로) 정의된 'top3_petal_length' 함수를 "적용(apply)"합니다.
# # 'apply'는 각 그룹에 대해 함수를 실행한 결과를 "하나의 데이터프레임"으로 결합하여 반환합니다.
# # 결과: 각 종(setosa, versicolor, virginica)별로 꽃잎 길이가 가장 긴 3개의 행(총 3 * 3 = 9행)이 추출되어 출력됩니다. MultiIndex(종, 원래 인덱스)가 생성됩니다.


# # 4. apply를 이용한 그룹 내 양자화 및 새로운 열 추가
# def q3cut(s):
#     # Series(s)를 3분위수(q=3)로 나누어 범주화(소, 중, 대)하고 문자열로 변환하여 반환합니다.
#     # qcut은 데이터의 분포에 따라 동일한 개수로 나누는 방식입니다.
#     return pd.qcut(s, 3, labels=["소", "중", "대"]).astype(str)

# print(iris.groupby(iris.species).petal_length.apply(q3cut))
# # 'petal_length' 열을 품종별로 그룹화한 후, 각 그룹 내에서만 독립적으로 3분위수로 나누어 범주화합니다.
# # 결과: (품종, 원본 인덱스)의 다중 인덱스를 가진, 범주화된 'petal_length' Series가 반환됩니다.

# print(q3cut(iris.petal_length))
# # 그룹화 없이 전체 'petal_length'에 대해 3분위수 범주화를 수행합니다.
# # 결과: 전체 데이터를 기준으로 범주화된 'petal_length' Series가 반환됩니다.

# print(iris.head(1)) # 데이터프레임의 첫 행을 출력합니다.
# print(iris.shape) # 데이터프레임의 크기(행, 열)를 출력합니다.

# print(iris.groupby(iris.species).petal_length.apply(q3cut))
# # (앞서 실행한 결과와 동일합니다. 품종별로 그룹 내 양자화 수행)

# iris['petal_length_class'] = iris.groupby(iris.species).petal_length.apply(q3cut).reset_index(drop=True)
# # 1. iris.groupby(iris.species).petal_length.apply(q3cut): 품종별로 독립적으로 범주화된 Series를 얻습니다.
# # 2. .reset_index(drop=True): 반환된 Series가 다중 인덱스(품종, 원본 인덱스)를 가지고 있으므로,
# #    이를 "원본 데이터프레임의 순서에 맞게" 단일 인덱스로 재설정하고 기존 인덱스(품종)는 버립니다.
# # 3. iris['petal_length_class'] = ...: 이 결과를 "원본 데이터프레임의 새로운 열"로 추가합니다.
# print(iris.head())
# # 결과: 새로운 'petal_length_class' 열이 추가된 데이터프레임의 상위 5개 행이 출력됩니다. 이 범주화는 "각 품종 내"의 상대적인 크기에 기반합니다.













# 그룹함수 및 피봇 테이블 이용 간단한 분석 예제
# 식당에서 식사 후 내는 팁(tip)과 관련된 데이터이용
# seaborn 패키지 내 tips 데이터셋 사용
#   total_bill: 식사대금
#   tip: 팁
#   sex: 성별
#   smoker: 흡연/금연 여부
#   day: 요일
#   time: 시간
#   size: 인원


# # tips 데이터셋 분석 주석
# tips = sns.load_dataset("tips") # Seaborn에서 'tips' 데이터셋을 불러와 데이터프레임 'tips'를 생성합니다.
# print(tips.tail()) # 데이터프레임의 마지막 5개 행을 출력합니다.

# # 'tip_pt' (Tip Percentage, 팁 비율) 열을 새로 생성합니다.
# # 'tip' 열의 값을 'total_bill' 열의 값으로 나누어 계산합니다.
# tips['tip_pt'] = tips['tip'] / tips['total_bill']
# print(tips.head()) # 새로운 열이 추가된 데이터프레임의 상위 5개 행을 출력합니다.
# print(tips.tail()) # 새로운 열이 추가된 데이터프레임의 마지막 5개 행을 출력합니다.


# # 1. 데이터프레임 정보 확인
# print(tips.info()) # 데이터프레임의 요약 정보(열 이름, Non-Null 값 개수, 데이터 타입, 메모리 사용량 등)를 출력합니다.

# print(tips.describe()) # 숫자형 열('total_bill', 'tip', 'tip_pt')에 대한 기술 통계(개수, 평균, 표준편차, 최솟값, 분위수, 최댓값)를 계산하여 출력합니다.

# print(tips) # 전체 데이터프레임을 출력합니다.


# # 2. groupby 객체 확인 및 그룹별 개수 계산
# test_df = pd.DataFrame(tips.groupby('sex', observed=False))
# # 'tips'를 'sex'를 기준으로 그룹화한 GroupBy 객체를 데이터프레임으로 변환합니다.
# # 열 0: 그룹 이름 ('Female', 'Male'), 열 1: 해당 그룹의 데이터프레임 객체
# print(test_df.loc[1][1].head(2))
# # 변환된 데이터프레임의 인덱스 1 (두 번째 그룹, 'Male')의 두 번째 열(데이터프레임 객체)에서 상위 2행을 출력합니다.
# print(test_df.loc[0][1].head(2))
# # 변환된 데이터프레임의 인덱스 0 (첫 번째 그룹, 'Female')의 두 번째 열(데이터프레임 객체)에서 상위 2행을 출력합니다.

# print(tips.groupby('sex', observed=False).count())
# # 'sex' 그룹별로 "결측값이 아닌" 데이터의 개수(Count)를 모든 열에 대해 계산합니다.
# # (결측값이 없으므로 .size()와 행 수가 같으나, .size()는 Series를 반환하고, .count()는 DataFrame을 반환합니다.)

# print(tips.groupby('sex', observed=False).size())
# # 'sex' 그룹별로 "각 그룹에 속하는 행의 총 개수"를 계산합니다. (Series 반환)

# t_gp = tips.groupby(['sex','smoker'], observed=False) # 'sex'와 'smoker' 두 개의 키를 기준으로 다중 그룹화 객체 't_gp'를 생성합니다.

# print(t_gp.size()) # 'sex' 및 'smoker'의 조합별 그룹에 속하는 행의 총 개수를 계산합니다. (Series 반환)



# # 3. 피벗 테이블을 이용한 그룹별 개수 계산
# print(tips.pivot_table(
#     'total_bill', # 값으로 사용할 열 (집계 대상)
#     index='sex', # 행 인덱스
#     columns='smoker', # 열 이름
#     aggfunc='count', # 집계 함수: 각 그룹의 개수(Count)
#     observed = False    
# ))
# # 결과: 성별('sex')과 흡연 여부('smoker') 조합별 청구서 금액('total_bill')의 개수, 즉 해당 그룹의 데이터 개수가 출력됩니다. (위의 t_gp.size()와 동일한 정보를 행렬 형태로 표현)



# # 4. 그룹별 평균 팁 비율 계산 및 비교
# print(tips.groupby('sex', observed=False)['tip_pt'].mean())
# # 'sex' 그룹별로 'tip_pt' 열의 평균을 계산합니다. (Series 반환)

# print(tips.groupby('smoker', observed=False)['tip_pt'].mean())
# # 'smoker' 그룹별로 'tip_pt' 열의 평균을 계산합니다. (Series 반환)

# print(tips.pivot_table(values='tip_pt', index='sex',columns='smoker', observed=False))
# # 피벗 테이블을 사용하여 'sex'와 'smoker' 그룹 조합별 'tip_pt'의 평균(기본 aggfunc='mean')을 계산합니다.



# # 5. 요일별 분석
# print(tips.groupby('day', observed=False).size())
# # 'day' (요일) 그룹별로 행의 총 개수를 계산합니다. (요일별 방문 횟수)

# print(len(tips)) # 전체 데이터프레임의 행 개수 (총 거래 건수)를 계산합니다.

# print(tips.groupby(['day','sex'], observed=False).size()/len(tips)*100)
# # 'day'와 'sex' 조합별 그룹 크기를 전체 거래 건수(len(tips))로 나누고 100을 곱하여 "전체에서 차지하는 백분율"을 계산합니다.

# print(tips.groupby('day', observed=False)['size'].mean()) # 이 코드는 오류가 발생합니다. GroupBy 객체는 'size'라는 열을 포함하고 있지 않습니다.

# print(tips.groupby('day', observed=False)['tip_pt'].mean())
# # 'day' 그룹별로 'tip_pt' 열의 평균을 계산합니다. (요일별 평균 팁 비율)

# print(tips.groupby('day', observed=False)[['total_bill','tip']].mean())
# # 'day' 그룹별로 'total_bill'과 'tip' 열 "두 개"의 평균을 계산합니다.



# # 6. 다중 그룹화 및 기술 통계
# print(tips.groupby(['day','sex','smoker'], observed=False)['tip_pt'].mean())
# # 'day', 'sex', 'smoker' 세 가지 키의 조합별로 'tip_pt'의 평균을 계산합니다.
# print(tips.groupby(['sex','smoker'], observed=False)[['tip_pt']].describe())
# # 'sex'와 'smoker' 두 가지 키의 조합별로 'tip_pt'에 대한 기술 통계('count', 'mean', 'std', 'min', 'max' 및 분위수)를 계산합니다. (DataFrame 반환)





























































































































