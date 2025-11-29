import matplotlib.pyplot as plt
import matplotlib as mpl
import korean
import pandas as pd
import numpy as np


"""
1. csv파일을 불러와서 데이터 프레임 생성
2. 구별 열을 인덱스로 설정
3. df컬럼명 "CCTV비율"에 "소계 / 인구수 * 100"로 value를 대입.
5. CCTV비율로 오름차순 정렬
6. CCTV비율로 오름차순 정렬한 데이터 프레임을 "cctv_ratio"변수에 대입함
7. 필요한 컬럼을 제외한 데이터만 버림.

"""


# 현재 실행 중인 파일의 전체 경로를 역슬래시("\\") 기준으로 분리하여 리스트로 만듭니다.
file_list = __file__.split("\\") 
# 파일 경로에서 리스트의 마지막 요소(파일 이름)를 "cctv_result.csv"로 대체하여
# 로드할 CSV 파일의 전체 경로를 만듭니다. (동일 디렉토리에 있다고 가정)
load_file = __file__.replace(file_list[-1], "cctv_result.csv")

# pandas 라이브러리를 사용하여 지정된 경로의 CSV 파일을 읽어와 DataFrame(df1)으로 저장합니다.
df1 = pd.read_csv(load_file)

# DataFrame의 인덱스를 기존 컬럼 '구별'로 설정합니다. (inplace=True로 원본 DataFrame 변경)
df1.set_index('구별', inplace=True)

# 'CCTV비율'이라는 새로운 컬럼을 생성합니다. 
# 계산은 '소계'(CCTV 수)를 '인구수'로 나눈 후 100을 곱하여 백분율로 구합니다.
df1['CCTV비율'] = df1['소계'] / df1['인구수'] * 100

# df1 DataFrame을 'CCTV비율' 컬럼을 기준으로 오름차순(ascending=True) 정렬하고,
# 결과를 "ratio_cctv" 변수에 할당합니다. 
# *주의: inplace=True로 원본 df1도 정렬되며, 이 경우 sort_values()는 None을 반환합니다.*
# ****정리: inplace=True를 사용하면 None값을 반환한다.
# 따라서 "ratio_cctv"는 "None"이 됩니다. 이 코드는 "plt.barh"에서 오류를 유발할 수 있습니다.
# (아래 "cctv_ratio" 변수를 사용하는 것이 일반적입니다.)
ratio_cctv = df1.sort_values(by='CCTV비율', ascending=True, inplace=True)

# df1 DataFrame을 'CCTV비율' 컬럼을 기준으로 오름차순 정렬하고, 
# "새로운" 정렬된 DataFrame을 "cctv_ratio" 변수에 할당합니다.
# (원본 df1은 이미 위 코드에서 정렬되었지만, 일반적으로 이렇게 별도 변수에 저장합니다.)
cctv_ratio = df1.sort_values(by='CCTV비율', ascending=True)



# Matplotlib을 사용하여 그래프를 그릴 준비를 합니다.
# 그래프 크기를 가로 14인치, 세로 10인치로 설정합니다.
plt.figure(figsize=(10,6))
# 수평 막대 그래프(barh)를 그립니다.
# x축 데이터는 "ratio_cctv"의 'CCTV비율' 컬럼, y축 데이터는 "ratio_cctv"의 인덱스('구별')를 사용합니다.
# *주의: ratio_cctv는 위에서 "None"이 되었으므로 이 부분에서 "AttributeError"가 발생합니다.*
# **올바른 사용을 위해서는 "cctv_ratio.index"와 "cctv_ratio['CCTV비율']"를 사용해야 합니다.**
plt.barh(cctv_ratio.index, cctv_ratio['CCTV비율'], color="#0F965D78", alpha=0.5, height=0.5)
# x축 레이블을 'CCTV비율'로 설정합니다.
plt.xlabel('CCTV비율')
# y축 레이블을 '구별'로 설정합니다.
plt.ylabel('구별')
# 그래프 제목을 'CCTV 비율'로 설정합니다.
plt.title('CCTV 비율')
# 생성된 그래프를 화면에 출력합니다.
plt.show()