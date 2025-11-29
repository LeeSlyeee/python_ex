import matplotlib.pyplot as plt
import matplotlib as mpl
import korean
import pandas as pd
import numpy as np

"""
##### 인구대비 cctv의 평균치를 확인하고 cctv가 과하게 부족한 구를 확인 #####

01. csv파일을 불러와서 데이터 프레임 생성.
02. 각 구별 인구대비 cctv평균치는 CCTV비율로 구해져 있다.
03. 각 구별 소계에 설치된 CCTV개수가 구해져 있다. 
04. CCTV의 평균치도 확인했는데 (1515.32) 가장 부족한 구는 어떻게 찾지?
05. 전체 소계를 돌면서 만약에 평균치 이하인 구는 평균치 - 소계로 갭차이를 구해서 가장 격차가 큰 구를 선정하면 될까? 
    - 개념없이 그냥 "소계"컬럼 잡아서 for문 돌렸더니 갭수치만 알고 해당하는 구는 알아낼 수 없었다.
06. 인구대비라는 말을 놓치고 작업했다. 처음부터 다시해야겠다.
07. 아니 근데 비율이 인구대비 CCTV비율이 아닌가??????
08. 그러면 인구대비 CCTV비율로 정렬하면 끝 아닌가?????
09. 가설1) CCTV비율이 가장 적은 구는 cctv가 과하게 부족한 구가 맞는가?

"""



file_list = __file__.split("\\") 
load_file = __file__.replace(file_list[-1], "cctv_result.csv")

# pandas 라이브러리를 사용하여 지정된 경로의 CSV 파일을 읽어와 DataFrame(df1)으로 저장합니다.
df1 = pd.read_csv(load_file)

# df1 DataFrame의 'CCTV비율' 컬럼에 대해 "mean()" 메서드를 사용하여 산술 평균을 계산하고,
# 그 결과를 "mean_cctv_ratio" 변수에 저장합니다.
mean_cctv_ratio = df1['CCTV비율'].mean()

# df1 DataFrame을 'CCTV비율' 컬럼을 기준으로 "내림차순(ascending=False)" 정렬합니다. 
# 즉, CCTV 비율이 높은 구가 상단에 위치하도록 정렬된 새로운 DataFrame을 "cctv_sorted" 변수에 할당합니다.
cctv_sorted = df1.sort_values(by='CCTV비율', ascending=False)

# Matplotlib을 사용하여 그래프를 그릴 준비를 합니다.
# 그래프 크기를 가로 10인치, 세로 10인치로 설정하여 "정사각형 형태"로 만듭니다.
plt.figure(figsize=(10, 10))

# 수평 막대 그래프(barh)를 그립니다.
# y축 데이터는 "cctv_sorted"의 '구별' 컬럼, x축 데이터는 'CCTV비율' 컬럼을 사용합니다.
# 정렬된 데이터 덕분에 막대가 비율이 높은 순서대로 나열됩니다.
plt.barh(cctv_sorted['구별'], cctv_sorted['CCTV비율'])

# 그래프에 "수직선(axvline)"을 추가합니다.
# x 위치는 앞서 계산한 "mean_cctv_ratio"(평균 CCTV 비율)로 설정합니다.
# 선의 색상은 'r'(빨간색), 스타일은 '--'(파선), 라벨은 평균값(소수점 둘째 자리까지)을 포함하여 설정합니다.
plt.axvline(x=mean_cctv_ratio, color='r', linestyle='--', label=f'평균 ({mean_cctv_ratio:.2f})')

# x축 레이블을 '인구 대비 CCTV 비율'로 설정합니다.
plt.xlabel('인구 대비 CCTV 비율')

# y축 레이블을 '구별'로 설정합니다.
plt.ylabel('구별')

# 그래프 제목을 '서울시 구별 인구 대비 CCTV 비율'로 설정합니다.
plt.title('서울시 구별 인구 대비 CCTV 비율')

# 범례(label로 설정한 평균선 설명)를 표시합니다.
plt.legend()

# x축 방향으로만 보조 그리드 선을 추가하여 값 비교를 "용이하게" 합니다.
# 투명도(alpha)는 0.5로 설정합니다.
plt.grid(axis='x', alpha=0.5)

# 생성된 그래프를 'cctv_analysis02.png' 파일로 "저장"합니다. (저장 후에도 그래프는 계속 존재합니다.)
plt.savefig('cctv_analysis02.png')
# 그래프 저장 완료 메시지를 출력합니다.
print("\n그래프가 'cctv_analysis02.png'로 저장되었습니다.")

# 생성된 그래프를 화면에 "출력 및 표시"합니다.
plt.show()



"""
가설이 맞은것 같다. 
acctv_analysis01.png에서 인구대비 cctv 비율을 구해 순위 비교와
acctv_analysis02.png의 데이터간의 상관관계가 있는 것으로 드러났다.

"""