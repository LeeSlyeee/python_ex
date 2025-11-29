import matplotlib.pyplot as plt
import matplotlib as mpl
import korean
import pandas as pd
import numpy as np

"""
##### 고령자 대비 cctv 비율 #####

01. 고령자 대비 CCTV 비율은 "소계 / 고령자 * 100"로 구할 수 있다.
02. 비율을 기준으로 내림차순 정렬을 한다.
03. 시각화한다. 

"""

file_list = __file__.split("\\") 
load_file = __file__.replace(file_list[-1], "cctv_result.csv")



# pandas 라이브러리를 사용하여 지정된 경로의 CSV 파일을 읽어와 DataFrame(df1)으로 저장합니다.
df1 = pd.read_csv(load_file)

# '고령자대비CCTV비율'이라는 "새로운 컬럼"을 생성합니다.
# 계산은 '소계'(CCTV 수)를 '고령자'(고령자 수)로 나눈 후 100을 곱하여 "백분율"로 구합니다.
# 이 비율은 고령자 100명당 CCTV가 몇 대 설치되어 있는지를 나타냅니다.
df1['고령자대비CCTV비율'] = df1['소계'] / df1['고령자'] * 100

# df1 DataFrame을 '고령자대비CCTV비율' 컬럼을 기준으로 "내림차순(ascending=False)" 정렬합니다.
# 즉, 고령자 대비 CCTV 비율이 높은 구가 상단에 위치하도록 정렬된 새로운 DataFrame을 "df1_sorted" 변수에 할당합니다.
df1_sorted = df1.sort_values(by='고령자대비CCTV비율', ascending=False)

# Matplotlib을 사용하여 그래프를 그릴 준비를 합니다.
# 그래프 크기를 가로 12인치, 세로 10인치로 설정합니다.
plt.figure(figsize=(12, 10))

# 수평 막대 그래프(barh)를 그립니다.
# y축 데이터는 "df1_sorted"의 '구별' 컬럼, x축 데이터는 '고령자대비CCTV비율' 컬럼을 사용합니다.
plt.barh(df1_sorted['구별'], df1_sorted['고령자대비CCTV비율'])

# x축 레이블을 '고령자 대비 CCTV 비율'로 설정합니다.
plt.xlabel('고령자 대비 CCTV 비율')

# y축 레이블을 '구별'로 설정합니다.
plt.ylabel('구별')

# 그래프 제목을 '서울시 구별 고령자 대비 CCTV 비율'로 설정합니다.
plt.title('구별 고령자 대비 CCTV 비율')

# x축 방향으로 보조 그리드 선을 추가하여 값 비교를 "용이하게" 합니다.
# 투명도(alpha)는 0.5로 설정합니다.
plt.grid(axis='x', alpha=0.5)

# 생성된 그래프를 'cctv_analysis03.png' 파일로 "저장"합니다.
plt.savefig('cctv_analysis03.png')
# 그래프 저장 완료 메시지를 출력합니다.
print("\n그래프가 'cctv_analysis03.png'로 저장되었습니다.")

# 생성된 그래프를 화면에 "출력 및 표시"합니다.
plt.show()



"""

"""