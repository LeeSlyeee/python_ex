import pandas as pd
import korean
import matplotlib.pyplot as plt
import numpy as np
korean.korean_setup()


file_list = __file__.split("\\") 
load_file = __file__.replace(file_list[-1], "notExercise.csv") 
df = pd.read_csv(load_file)

input_category = input("대분류 이름은?")

def get_data_dict(chart, cate):
    # [DataFrame 필터링] 원본 DataFrame인 'chart'에서 특정 조건에 맞는 행들만 추출하여 새로운 DataFrame 'data_df'를 생성합니다.
    #    - chart['대분류'] == cate: 'chart' DataFrame의 '대분류'라는 컬럼(열)의 값이
    #      외부에서 입력된 변수 'cate'의 값과 일치하는 모든 행에 대해 True/False로 구성된 불리언(Boolean) Series를 생성합니다.
    #    - chart[...] : 이 불리언 Series를 필터링 마스크로 사용하여, True인 행들만 선택합니다.
    # [결과] 'data_df'는 사용자가 지정한 '대분류'에 속하는 데이터만 담고 있는 작은 DataFrame이 됩니다.
    data_df = chart[chart['대분류'] == cate]
    plt.figure(figsize=(10, 10))
    
    # 데이터 컬럼 인덱스 정의 (3번째부터 6번째 컬럼까지)
    # data_df.columns[3:7]은 ["컬럼3", "컬럼4", "컬럼5", "컬럼6"] 이름 리스트를 반환합니다.
    column_names = data_df.columns[3:7]
    
    for i in range(len(column_names)):
        # 4. 2행 2열 중 i+1번째(1, 2, 3, 4) 위치에 subplot 생성
        plt.subplot(2, 2, i + 1)
        
        # 5. 제목 설정
        plt.title(
            column_names[i],
            fontsize=14,
            color="#303030",
            weight='bold'
        )
        
        # plt.pie()에 전달되는 data_df[column_names[i]]는 Series 객체입니다.
        # 만약 '대분류' 필터링 후에도 행이 여러 개라면, 해당 컬럼의 모든 값이 파이 조각이 됩니다.
        plt.pie(
            data_df[column_names[i]], # 대분류로 정제된 데이터 프레임에서 특정 컬럼값만 가져옴.
            labels=data_df["분류"],
            autopct='%.1f%%',
            startangle=200,
            counterclock=False
        )
        
    plt.tight_layout()
    plt.show()
    
# get_data_dict(df, input_category)