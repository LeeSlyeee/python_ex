import pandas as pd
import numpy as np
import time
import seaborn as sns




# # # 데이터 개수 세기
# # --- Series 생성 및 결측값 할당 ---
# s = pd.Series(range(10)) # pd.Series()를 호출하여 0부터 9까지의 정수(총 10개)를 갖는 Series 's'를 생성합니다.
#                          # 인덱스는 기본적으로 0부터 9까지의 정수가 됩니다.

# s[3] = np.nan # Series 's'의 "인덱스 3" 위치에 해당하는 값에 "np.nan"을 할당합니다.
#               # np.nan은 "Not a Number"를 의미하며, 데이터 분석에서 결측값(Missing Value)을 나타낼 때 사용됩니다.
#               # 이 작업으로 인해 정수형이었던 Series의 dtype(데이터 타입)이 NaN을 포함할 수 있는 "float64"로 자동 변경될 수 있습니다.

# print(s)      # 결측값이 포함된 Series 's'의 내용과 인덱스, 데이터 타입을 출력합니다.


# # --- 유효값 개수 세기 ---
# print(s.count())    # Series의 `.count()` 메서드를 호출하여 결과를 출력합니다.
#                     # 이 메서드는 Series 내에서 "유효한 값(숫자, 문자열 등)"의 개수를 세고,
#                     # "결측치 (NaN, None 등)"는 "계산에서 제외"합니다.
#                     # 원래 10개의 값 중 1개가 NaN이 되었으므로, 9가 출력될 것입니다.
                    
                    
                    
                    
# # # 난수 발생시켜 dataframe 생성
# # --- 단일 난수 생성 ---
# print(np.random.randint(5)) # np.random 모듈의 .randint() 함수를 호출합니다.
#                             # 인수가 하나(5)만 있을 경우, 0(포함)부터 5(미포함) 사이의 "단일 정수"를 무작위로 생성하여 반환합니다.
#                             # 즉, 0, 1, 2, 3, 4 중 하나가 출력됩니다.
#                             # (실행할 때마다 다른 값이 나옵니다.)


# np.random.seed(3)  # NumPy의 난수 생성기(random number generator)에 "시드 값 3"을 설정합니다.
#                    # "'시드(Seed)'"는 난수 생성 알고리즘이 시작하는 기준점 역할을 하는 정수입니다.             
#                    # "난수(Random Number)의 특징":
#                    # 1. 이 코드를 실행한 후 다음에 생성되는 모든 난수(예: np.random.rand(), np.random.randint() 등)의
#                    #    "순서와 값은 항상 고정"됩니다.
#                    # 2. 프로그램이 다시 실행되어 이 `np.random.seed(3)` 코드를 만나면,
#                    #    이전에 생성했던 난수 순서와 "정확히 동일한 난수 순서"가 다시 시작됩니다.
#                    # 3. 만약 시드를 설정하지 않거나 다른 시드 값(예: 4)을 사용했다면,
#                    #    실행할 때마다 다른 난수 시퀀스가 생성됩니다.


# # --- 다중 난수 배열 생성 ---
# print(np.random.randint(5,       # 난수 생성 범위의 "상한선(High)"을 5(미포함)로 지정합니다. (0부터 4까지)
#                   size = 4))     # "size" 인자를 사용하여 생성할 난수의 "개수"를 지정합니다.
#                                  # 이 함수는 0부터 4 사이의 정수 4개를 무작위로 포함하는 "1차원 NumPy 배열"을 반환합니다.
#                                  # (예시 결과: [3 0 4 1] 와 같이 4개의 정수로 구성된 배열이 나옵니다.)
                            

# print(np.random.randint(5, size = 4))   # 다시 0부터 5 미만의 정수 중 난수를 4개 생성합니다.
#                                         # 시드가 이미 설정되었기 때문에, 이 배열은
#                                         # 앞서 생성된 난수 다음 순서의 4개 난수로 구성됩니다.
#                                         # !! 주의 !!: 이 코드만으로는 두 결과가 같지 않습니다.
#                                         # 시드를 재설정하지 않았기 때문에, 첫 번째 호출 이후
#                                         # 난수 생성기의 "상태(state)"가 변경되어 다음 난수 시퀀스가 이어져 나옵니다.
                               
                               






# # --- 1. 시드 설정 (매번 다른 난수 시퀀스 생성) ---
# # time.time()은 1970년 1월 1일 0시 0분 0초(UTC) 이후 경과된 초를 부동 소수점 형태로 반환합니다.
# # int(...)는 이 부동 소수점을 정수로 변환하여 시드 값으로 사용합니다.
# # np.random.seed(...)는 난수 생성기의 시작점을 설정합니다.
# np.random.seed(int(time.time())) 
#                   # 시드가 "실행 시점의 현재 시간"을 기반으로 설정되므로,
#                   # 코드를 실행할 때마다 시드 값이 달라지고, 그 결과 생성되는
#                   # "난수 시퀀스는 매번 다르게" 됩니다. (재현성 확보 X, 무작위성 확보 O)

# # --- 2. 난수 생성 및 출력 ---
# print(np.random.randint(5, size=4)) # 0 (포함)부터 5 (미포함) 사이의 정수 중 난수를 
#                                     # 4개(size=4) 생성하여 1차원 배열로 반환합니다.
#                                     # 이 결과는 실행할 때마다 다른 배열이 됩니다.

# # --- 3. 2차원 난수 배열 생성 ---
# print(np.random.randint(5, size=(4,4))) # 0부터 5 미만의 정수 중 난수를 생성합니다.
#                                         # size=(4, 4)는 생성된 난수들을 "4행 4열의 2차원 배열"로 만들도록 지정합니다.
#                                         # (총 16개의 난수가 생성됩니다.)








# # # --- 1. 난수 시드 설정 ---
# np.random.seed(3) # NumPy의 난수 생성기 "시드(seed)"를 "3"으로 고정합니다.
#                   # 이렇게 하면 이 코드를 실행할 때마다 "동일한 난수 시퀀스"가 생성되어 결과의 "재현성"이 보장됩니다.

# # # --- 2. DataFrame 생성 ---
# # # np.random.randint(5, size = (4, 4)): 0부터 5 미만(즉, 0, 1, 2, 3, 4)의 정수 난수를
# # #                                      4행 4열의 2차원 배열로 생성합니다.
# df1 = pd.DataFrame(np.random.randint(5, size = (4, 4))) # 생성된 배열을 데이터로 사용하여 DataFrame df1을 생성합니다.
#                                                         # 열 이름(0, 1, 2, 3)과 행 이름(0, 1, 2, 3)은 기본 정수가 사용됩니다.

# # --- 3. 특정 값 결측치로 변경 ---
# # .iloc[행 인덱스, 열 인덱스]: "정수 위치(순서)"를 사용하여 DataFrame의 특정 셀에 접근합니다.
# # df1.iloc[2, 3]: "3번째 행" (인덱스 2)의 "4번째 열" (인덱스 3)에 해당하는 셀을 선택합니다.
# df1.iloc[2,3] = np.nan # 선택한 셀의 값을 "np.nan (Not a Number, 결측치)"으로 변경합니다.


# # --- 4. 유효값 개수 세기 ---
# print(df1.count())      # DataFrame의 `.count()` 메서드를 호출하여 결과를 출력합니다.
#                         # 이 메서드는 "각 열(Column)"에 대해 "결측치(NaN)를 제외한" 유효한 값의 개수를 셉니다.
#                         # 모든 열은 원래 4개의 값을 가졌지만, 3번 열의 값이 NaN으로 바뀌었으므로,
#                         # 3번 열의 카운트는 3이 되고 나머지 열은 4가 될 것입니다.


# # --- 5. 결과 출력 ---
# print(df1) # 최종적으로 수정된 DataFrame df1을 출력합니다.


# --- 6. 값 정렬 시도 ---
# print(df1.sort_values()) # DataFrame에 대해 `.sort_values()` 메서드를 호출하고 그 결과를 출력합니다.
#                          # DataFrame에서 `by` 인자 없이 `.sort_values()`를 호출하면, 
#                          # "기본적으로 모든 열(Column)을 기준으로 정렬"을 시도합니다.
#                          # 이는 "모호한(Ambiguous)" 정렬 기준이 되므로, 
#                          # pandas는 정렬 기준으로 사용할 열 이름(by='열 이름')을 "지정하도록 요구"하는 오류를 발생시킵니다.
#                          # 일반적으로 "DataFrame 정렬 시에는 `by` 인자를 반드시 명시"해야 합니다.
                         

# # --- 7. 오름차순 정렬 ---
# print(df1.sort_values(by=0)) # DataFrame df1을 `.sort_values()` 메서드를 사용하여 정렬합니다.
#                              # "by=0"은 "열 이름이 0인 열"의 값을 기준으로 정렬하라는 의미입니다.
#                              # `ascending` 옵션을 지정하지 않았으므로, 기본값인 "오름차순 (작은 값 -> 큰 값)"으로 정렬됩니다.
#                              # 정렬 결과, "행 전체"가 기준 열(0번 열)의 순서에 따라 재배치됩니다.

# # --- 8. 내림차순 정렬 ---
# print(df1.sort_values(by=0, # 정렬 기준 열은 동일하게 "열 0"입니다.
#                       ascending=False)) # "ascending=False" 옵션을 사용하여
#                                         # "내림차순 (큰 값 -> 작은 값)"으로 정렬합니다.


# # --- 9. 정렬 및 출력 ---
# print(df1.sort_values(
#     by=[0, 2],         # 정렬 기준으로 사용할 열(Column) 이름의 리스트를 지정합니다. 
#                        # "1차적으로 '0'번 열"을 기준으로 정렬하고, 
#                        # "'0'번 열의 값이 같으면 2차적으로 '2'번 열"을 기준으로 정렬합니다.
#     ascending=False    # 정렬 방식을 지정합니다. "False"는 "내림차순 (Descending)"을 의미합니다. 
#                        # (가장 큰 값부터 작은 값 순서로 정렬)
# )) # 정렬된 DataFrame을 출력합니다.







# # --- 1. DataFrame 생성 ---
# titanic = pd.read_csv('./data/test.csv') # './data/test.csv' 경로의 CSV 파일을 읽어와 
#                                         # titanic 변수에 DataFrame으로 저장합니다.
#                                         # 이 파일은 타이타닉 승객 정보를 담고 있는 데이터일 것입니다.

# # --- 2. 열 삭제 ---
# del titanic['Unnamed: 0'] # DataFrame 'titanic'에서 "'Unnamed: 0'"이라는 이름의 열(Column)을 "삭제"합니다.
#                          # 이 열은 보통 CSV 파일을 저장할 때 잘못 포함된 이전 인덱스 값 등을 담고 있는 불필요한 열인 경우가 많습니다.
#                          # 'del' 키워드를 사용하면 "원본 DataFrame"에서 해당 열이 영구적으로 제거됩니다.

# # --- 3. 상위 행 확인 ---
# print(titanic.head(10)) # DataFrame 'titanic'의 "가장 첫 10개 행(Row)"을 출력합니다.
#                         # head()는 기본적으로 5개 행을 보여주지만, 10을 인수로 지정하여 출력 행 수를 늘렸습니다.
#                         # 이는 데이터의 구조와 상위 레코드를 빠르게 확인하기 위해 사용됩니다.

# # --- 4. 하위 행 확인 ---
# print(titanic.tail(2))  # DataFrame 'titanic'의 "가장 마지막 2개 행(Row)"을 출력합니다.
#                         # tail()은 기본적으로 5개 행을 보여주지만, 2를 인수로 지정하여 출력 행 수를 줄였습니다.
#                         # 이는 데이터의 마지막 부분이 어떻게 구성되어 있는지 확인하기 위해 사용됩니다.
                        
# # --- 5. 수정된 데이터 저장 ---
# titanic.to_csv('./data/test.csv') # 수정된 DataFrame 'titanic'을 "원래 파일 경로"인 './data/test.csv'에 "덮어쓰기"하여 저장합니다.
#                                   # 이 저장 과정에서 DataFrame의 행 인덱스(0, 1, 2, ...)도 파일에 새로운 열로 함께 저장됩니다.
#                                   # 만약 인덱스를 저장하지 않으려면 `index=False` 옵션을 추가해야 합니다.
                                  
# # --- 6. 'alive' 열 분석 ---
# # 6-1. 데이터 타입 확인
# print(titanic['alive'].dtype)            # DataFrame 'titanic'에서 'alive'라는 이름의 열(Series)을 선택한 후,
#                                          # 해당 열의 "데이터 타입(dtype)"을 출력합니다.
#                                          # 'alive'는 생존 여부(예: 'yes', 'no' 또는 1, 0)를 나타내는 문자열 또는 정수 타입일 것입니다.

# # 6-2. 절대 빈도수 계산
# print(titanic['alive'].value_counts())   # 'alive' 열의 고유한 값들(예: 'yes', 'no')이 각각 "몇 번" 나타났는지
#                                          # (빈도수 또는 개수)를 계산하여 출력합니다.
#                                          # 결과는 빈도수가 높은 순으로 정렬된 Series 형태로 나옵니다.

# # 6-3. 상대 빈도수 (비율) 계산
# print(titanic['alive'].value_counts(normalize=True)) # `.value_counts()` 메서드에 "normalize=True" 옵션을 추가하여
#                                                      # 각 값의 "전체 데이터 대비 비율 (상대 빈도수)"을 계산하여 출력합니다.
#                                                      # 이는 생존율 또는 사망률을 파악하는 데 유용합니다.

# # --- 특정 열의 비율 계산 및 출력 ---
# print( # 계산된 결과를 출력합니다.
#     titanic['alive'] # DataFrame 'titanic'에서 "'alive'"라는 이름의 열(Column, Series)을 선택합니다.
#                      # 이 열은 탑승객의 생존 여부(예: 'yes', 'no' 또는 1, 0)를 담고 있을 것으로 예상됩니다.
#     .value_counts(normalize=True) # 선택된 'alive' 열의 "고유한 값들" (예: 'yes'와 'no')에 대해 빈도수를 계산합니다.
#                                   # 이때, "normalize=True" 옵션을 사용하여 "절대 빈도수"가 아닌
#                                   # "전체 데이터 수 대비 비율 (상대 빈도수)"을 계산합니다. (결과: 0과 1 사이의 실수)
#     * 100 # 계산된 비율(0.xx)에 "100을 곱하여" 결과를 백분율(퍼센트, %) 형식으로 변환합니다.
# ) # 최종적으로 생존 여부별 비율(%)이 Series 형태로 출력됩니다.


# # --- 절대 빈도수 계산 및 출력 ---
# print(titanic['sex'].value_counts())     # DataFrame 'titanic'의 "'sex' 열(Column)"을 선택한 후,
#                                          # `.value_counts()` 메서드를 호출하여 "각 고유값(예: 'male', 'female')"의
#                                          # "절대 빈도수 (개수)"를 계산하고 출력합니다.
#                                          # 결과는 빈도수가 높은 순서로 정렬된 Series 형태로 나옵니다.

# # --- 상대 빈도수 (백분율) 계산 및 출력 ---
# print(titanic['sex'].value_counts(normalize=True) # `normalize=True` 옵션을 사용하여 "상대 빈도수 (전체 대비 비율)"를 계산합니다. 
#       * 100)                                       # 비율 값(0과 1 사이의 실수)에 "100을 곱하여 백분율(%)"로 변환한 후 출력합니다.


# # --- 특정 열의 조합 빈도 계산 및 출력 ---
# # titanic[['sex','alive']]: DataFrame 'titanic'에서 "'sex'" 열과 "'alive'" 열, 두 개를 선택하여 
# #                           새로운 DataFrame (또는 2열짜리 Series)을 만듭니다. (이중 대괄호 사용)
# print(titanic[['sex','alive']].value_counts()) # 선택된 두 열에 대해 `.value_counts()` 메서드를 호출합니다.
#                                                # 이 메서드는 'sex'와 'alive'의 "모든 고유한 조합" (예: 남성 & 생존, 여성 & 사망 등)
#                                                # 각각이 데이터에 몇 번 나타났는지 (빈도수)를 계산하여 출력합니다.
#                                                # 결과는 빈도수가 높은 순서(내림차순)로 정렬됩니다.


# # --- 값의 빈도수 계산 및 타입 확인 ---
# # titanic[['sex','alive']]: 
# #   DataFrame 'titanic'에서 "'sex'와 'alive' 두 개의 열"을 선택합니다. 
# #   이 때 결과는 "새로운 DataFrame" 형태(두 개의 열을 가진)가 됩니다.

# # .value_counts(): 
# #   선택된 두 열(DataFrame)에 대해 `.value_counts()` 메서드를 호출합니다.
# #   이 메서드는 두 열의 "모든 고유한 조합"('sex'와 'alive'의 페어) 각각이 데이터에 몇 번 나타났는지(빈도수)를 계산합니다.
# #   이 메서드의 반환 값은 항상 "pandas Series" 객체입니다. (인덱스는 고유한 조합을 나타내는 MultiIndex입니다.)

# print(type(titanic[['sex','alive']].value_counts())) # 계산된 빈도수의 결과 객체 "타입"을 출력합니다.
#                                                      # 결과는 `<class 'pandas.core.series.Series'>`가 될 것입니다.
 

# # --- 값의 빈도수 계산 및 DataFrame 변환 ---
# # titanic[['sex', 'alive']]: 원본 DataFrame 'titanic'에서 'sex'와 'alive' 두 열만 선택하여 새로운 DataFrame을 생성합니다.
# # .value_counts(): 선택된 두 열의 "모든 행 조합"에 대해 "빈도수(count)"를 계산합니다.
# #                  (예: 'male'이면서 'yes'인 경우가 몇 번인지 등)
# #                  결과는 MultiIndex를 가진 pandas Series로 반환됩니다.
# print(pd.DataFrame(titanic[['sex','alive']].value_counts())) 
#                                          # 앞서 계산된 Series 형태의 빈도수 결과를
#                                          # 다시 "DataFrame"으로 변환합니다.
#                                          # 이 DataFrame은 'sex'와 'alive'의 조합을 인덱스로,
#                                          # 빈도수를 값(count)으로 하는 단일 열을 갖게 됩니다.
#                                          # (변수에 할당되지 않았으므로, 최종 결과가 출력됩니다.)












                                  
                                  
                                  
# # --- 1. Series 생성 ---
# np.random.seed(1)  # 난수 생성기의 시드(Seed)를 "1"로 고정합니다.
#                    # 이를 통해 다음에 생성되는 난수 시퀀스가 항상 동일하게 유지되어 결과를 재현할 수 있습니다.

# # np.random.randint(6, size=100): 0 (포함)부터 6 (미포함) 사이의 정수 난수를 100개 생성합니다.
# s2=pd.Series(np.random.randint(6,size=100)) # 이 100개의 난수 배열을 데이터로 사용하여
#                                             # pandas Series 객체 's2'를 생성하고 변수에 할당합니다.
#                                             # (인덱스는 0부터 99까지의 정수로 자동 설정됩니다.)

# # --- 2. 데이터 확인 및 출력 ---
# print(s2.tail())   # Series의 "마지막 5개 행(row)"을 출력합니다.
#                    # `.tail()` 메서드는 기본적으로 마지막 5개의 데이터를 보여줍니다.
                   
# print(s2.head())   # Series의 "첫 5개 행(row)"을 출력합니다.
#                    # `.head()` 메서드는 기본적으로 첫 5개의 데이터를 보여줍니다.
                   
# print(s2.head(10)) # Series의 `.head(10)` 메서드를 사용하여 "첫 10개 행"을 출력합니다.
                   
# print(len(s2))     # 파이썬의 `len()` 함수를 사용하여 Series 's2'에 담긴 "총 데이터 개수"를 출력합니다.
#                    # (여기서는 100이 출력됩니다.)
                   


# # --- 3. 절대 빈도수 계산 ---
# print(s2.value_counts()) # Series s2에 대해 `.value_counts()` 메서드를 호출합니다.
#                          # 이 메서드는 Series의 "고유한 값(Unique Value)" 각각이 "몇 번 나타났는지 (빈도수)"를 계산하여 반환합니다.
#                          # 결과는 빈도수가 높은 순서(내림차순)로 정렬된 새로운 Series 형태로 나옵니다.

# # --- 4. 상대 빈도수 (비율) 계산 ---
# print(s2.value_counts(normalize=True))  # `.value_counts()` 메서드를 호출할 때 "normalize=True" 옵션을 추가합니다.
#                                         # 이 옵션은 빈도수 대신 "전체 데이터 수 대비 해당 값의 비율 (상대 빈도수)"을 계산하여 반환합니다.
#                                         # 각 값은 0과 1 사이의 실수(float)로 표시됩니다.


# # --- 5. 빈도수 계산 및 인덱스 정렬 후 출력 ---
# # s2.value_counts(): Series s2의 "고유한 값(Unique Value)" 각각의 "빈도수"를 계산하여 반환합니다.
# #                    결과는 빈도수가 높은 순서(내림차순)로 정렬된 새로운 Series 형태입니다.
# # .sort_index(): 앞서 계산된 Series의 "인덱스(여기서는 고유한 값 0, 1, 2, 3, 4, 5)"를 기준으로 "오름차순 정렬"합니다.
# #                (기본 .value_counts()는 빈도수 기준으로 정렬되는데, 이 코드는 값 순서대로 정렬하도록 변경합니다.)
# print(s2.value_counts().sort_index()) # 정렬된 결과를 콘솔에 출력합니다.


# # --- 6. 빈도 계산 및 인덱스 역순 정렬 후 출력 ---
# # s2.value_counts(): Series s2의 "고유한 값" 각각이 몇 번 나타났는지(빈도수)를 계산하여 Series로 반환합니다.
# #                    기본적으로 빈도수가 높은 순서(내림차순)로 정렬됩니다.
# # .sort_index(ascending=False): 앞서 계산된 value_counts의 결과(Series)를 "인덱스(여기서는 0~5의 값)"를 기준으로 정렬합니다.
# #                               "ascending=False" 옵션은 내림차순 정렬(큰 값 -> 작은 값)을 의미합니다.
# print(s2.value_counts().sort_index(ascending=False)) # 최종 결과를 콘솔에 출력합니다.


# # --- 7. 빈도수 계산, 정렬 및 출력 ---
# # s2.value_counts(): Series s2의 "고유한 값" 각각이 "몇 번 나타났는지 (빈도수)"를 계산합니다.
# #                    결과는 빈도수가 높은 순서(내림차순)로 정렬된 새로운 Series 형태로 반환됩니다.
# # .sort_values(): 앞서 계산된 빈도수 Series에 대해 이 메서드를 호출합니다.
# #                 기본적으로 "오름차순"으로 정렬됩니다 (가장 작은 빈도수부터).
# #                 만약 내림차순으로 정렬하려면 `ascending=False` 옵션을 사용해야 합니다.
# print(s2.value_counts().sort_values()) # 빈도수가 오름차순으로 정렬된 최종 Series를 출력합니다.


# # --- 8. 빈도수 계산 및 출력 ---
# # s2.value_counts(): Series s2의 "고유한 값" 각각이 "몇 번 나타났는지 (빈도수)"를 계산하여 새로운 Series를 반환합니다.
# #                    (기본적으로는 이미 빈도수 기준으로 내림차순 정렬되어 있습니다.)
# # .sort_values(ascending=False): 앞서 계산된 빈도수(Series의 값)를 기준으로 "내림차순" (가장 큰 값부터)으로 명시적으로 다시 정렬합니다.
# #                                (여기서 ascending=False는 내림차순을 의미하며, value_counts()의 기본값과 동일합니다.)
# print(s2.value_counts().sort_values(ascending=False)) # 최종적으로 빈도수 기준으로 내림차순 정렬된 결과를 출력합니다.














# # --- 1. DataFrame 생성 ---
# df = pd.DataFrame({ # DataFrame을 생성하고 그 결과를 변수 df에 저장합니다.
#     'num_legs': [2, 4, 4, 6], # 'num_legs' (다리 수)라는 열(Column)의 데이터: [2, 4, 4, 6]
#     'num_wings': [2, 0, 0, 0] # 'num_wings' (날개 수)라는 열(Column)의 데이터: [2, 0, 0, 0]
# }, index=['falcon', 'dog', 'cat', 'ant']) # index 매개변수를 사용하여 행(Row)의 레이블(이름)을 
#                                           # ['falcon', 'dog', 'cat', 'ant']으로 설정합니다.
                                          

# # --- 2. 인덱스 기준 정렬 ---
# print(df.sort_index()) # DataFrame의 "행 인덱스"를 기준으로 정렬합니다.
#                        # 인자가 없으므로 기본값인 "오름차순 (ascending=True)"으로 정렬합니다.
#                        # 인덱스가 문자열이므로 알파벳 순서(ant, cat, dog, falcon)로 정렬됩니다.

# print(df.sort_index(ascending=False)) # DataFrame의 "행 인덱스"를 기준으로 정렬합니다.
#                                       # "ascending=False" 인자를 주어 "내림차순"으로 정렬합니다.
#                                       # 알파벳 역순(falcon, dog, cat, ant)으로 정렬됩니다.
                                      
                                      











# # --- 1. 시드 설정 및 DataFrame 생성 ---
# np.random.seed(1) # 난수 생성기의 "시드(Seed)"를 "1"로 고정합니다.
#                   # 이는 다음에 생성되는 모든 난수 시퀀스가 항상 동일하도록 보장합니다 (결과 재현성).

# # np.random.randint(10, size=(4,8)): 0 (포함)부터 10 (미포함, 즉 9) 사이의 정수 난수(총 32개)를
# #                                     "4행 8열의 2차원 배열"로 생성합니다.
# df2=pd.DataFrame(np.random.randint(10,size=(4,8))) # 이 4x8 난수 배열을 데이터로 갖는 DataFrame 'df2'를 생성합니다.
#                                                    # (열 이름은 0부터 7, 행 인덱스는 0부터 3까지의 정수로 기본 설정됩니다.)

# # --- 2. 행별 합계 계산 (axis=1) ---
# print(df2.sum(axis=1)) # `.sum()` 메서드에 "axis=1"을 지정하여 "행(Row)"을 따라 합계를 계산합니다.
#                        # 즉, 각 행의 모든 열 값을 더한 "결과(합계)"가 새로운 Series로 반환됩니다.
#                        # (결과: 0행 합계, 1행 합계, 2행 합계, 3행 합계)

# # --- 3. 열별 합계 계산 (axis=0) ---
# print(df2.sum(axis=0)) # `.sum()` 메서드에 "axis=0"을 지정하여 "열(Column)"을 따라 합계를 계산합니다.
#                        # 즉, 각 열의 모든 행 값을 더한 "결과(합계)"가 새로운 Series로 반환됩니다.
#                        # (결과: 0열 합계, 1열 합계, ..., 7열 합계)

# # --- 4. 기본 동작 (axis 생략) ---
# print(df2.sum())       # `.sum()` 메서드에 "axis 인자가 생략"되었습니다.
#                        # pandas의 기본 동작은 "axis=0"으로, "열(Column)"을 따라 합계를 계산합니다.
#                        # 따라서, 이 결과는 바로 위 `df2.sum(axis=0)`의 결과와 "동일"합니다.


# # --- 5. 'total' 열 추가 ---
# # df2[[0,1,2,3,4,5,6,7]]: df2의 모든 열(0부터 7까지)을 선택합니다.
# # .sum(axis=1): 선택된 열들에 대해 "axis=1 (행 기준)"으로 합계를 계산합니다.
# #              즉, 각 행의 모든 값(8개)을 더합니다.
# df2['total'] = df2[[0,1,2,3,4,5,6,7]].sum(axis=1) # 계산된 행별 합계 결과를 'total'이라는 새 열에 추가합니다.
# # # df2의 모양: 4행 9열 (0~7 열, total 열)


# # --- 6. 통계 연산 및 출력 ---
# # 통계 연산 메서드(mean, min, max 등)는 기본적으로 axis=0 (열 기준)으로 작동합니다.

# print(df2.mean(axis=0)) # DataFrame의 "각 열(Column, axis=0)"에 대해 "평균"을 계산하여 출력합니다.(총 4개 행의 값들의 평균이 계산됩니다.)
# print(df2.mean(axis=1)) # DataFrame의 "각 행(Row, axis=1)"에 대해 "평균"을 계산하여 출력합니다.(총 9개 열의 값들(0~7, total)의 평균이 계산됩니다.)

# print(df2.min(axis=0))  # DataFrame의 "각 열(Column, axis=0)"에서 "최솟값"을 찾아 출력합니다.
# print(df2.min(axis=1))  # DataFrame의 "각 행(Row, axis=1)"에서 "최솟값"을 찾아 출력합니다.

# print(df2.max(axis=0))  # DataFrame의 "각 열(Column, axis=0)"에서 "최댓값"을 찾아 출력합니다.
# print(df2.max(axis=1))  # DataFrame의 "각 행(Row, axis=1)"에서 "최댓값"을 찾아 출력합니다.



# # --- 7. 'max_data' 행 추가 (열 최댓값) ---
# # df2.loc['max_data']: 'max_data'라는 레이블을 가진 새로운 행을 선택하거나 생성합니다.
# # .max(axis=0): 0번 열부터 7번 열까지의 값에 대해 "세로 방향 (열 방향, axis=0)"으로 최댓값을 계산합니다.
# df2.loc['max_data']=df2[[0,1,2,3,4,5,6,7]].max(axis=0) # 계산된 열 최댓값들을 'max_data'라는 새 행으로 df2에 추가합니다.
# # df2는 이제 5행 9열이 됩니다.


# # --- 8. 'total' 열 삭제 (inplace=True 사용) ---
# df2.drop('total',  # 삭제할 대상의 레이블('total' 열)을 지정합니다.
#          axis=1,   # "axis=1"은 "열(Column)"을 삭제하라는 의미입니다.
#          inplace=True) # "inplace=True"는 "원본 DataFrame (df2) 자체를 변경"하도록 지정합니다.
# print(f"df2.drop('total',axis=1,inplace=True) >>>> {df2}") # 'total' 열이 삭제된 df2의 내용이 출력됩니다.
# # df2는 이제 5행 8열이 됩니다.


# # --- 9. 'max_data' 행 삭제 (inplace=False, 기본값 사용) ---
# df2.drop('max_data', # 삭제할 대상의 레이블('max_data' 행)을 지정합니다.
#          axis=0)     # "axis=0"은 "행(Row)"을 삭제하라는 의미입니다.
#                      # "inplace 옵션이 없거나 False이므로", 이 연산은 'max_data' 행이 제거된 "새로운 DataFrame을 반환"하지만, 
#                      # 그 결과를 변수에 저장하지 않아 "원본 df2에는 아무런 변화가 없습니다."
# print(f"df2.drop('max_data',axis=0) >>>> {df2}") # 원본 df2에는 'max_data' 행이 여전히 포함되어 있는 상태로 출력됩니다.


# # --- 10. 결측치 삽입 ---
# df2.iloc[0,0] = np.nan # .iloc를 사용하여 "첫 번째 행(인덱스 0)", "첫 번째 열(인덱스 0)" 위치에
#                        # "NaN (결측치)" 값을 할당합니다.
#                        # (이는 DataFrame에 결측치가 존재하는지 테스트하기 위함입니다.)


# # --- 11. 결측치가 포함된 행 삭제 시도 (원본 변경 없음) ---
# df2.dropna(axis=0) # "axis=0" (행)을 기준으로 결측치가 있는 행을 삭제합니다.
#                    # df2.iloc[0,0]에 의해 첫 번째 행(인덱스 0)에 NaN이 있으므로, 이 행이 제거된 "새로운 DataFrame"을 반환합니다.
# print(f"df2.dropna(axis=0) >>>> {df2}") # "inplace=False가 기본값"이므로, "원본 df2는 변경되지 않습니다."


# # --- 12. 결측치가 포함된 열 삭제 시도 (원본 변경 없음) ---
# df2.dropna(axis=1) # "axis=1" (열)을 기준으로 결측치가 있는 열을 삭제합니다.
#                    # df2.iloc[0,0]에 의해 첫 번째 열(인덱스 0)에 NaN이 있으므로, 이 열이 제거된 "새로운 DataFrame"을 반환합니다.
# print(f"df2.dropna(axis=1) >>>> {df2}") # "inplace=False가 기본값"이므로, "원본 df2는 변경되지 않습니다."


# # --- 13. 결측치가 포함된 열 삭제 및 원본 반영 ---
# df2.dropna(axis=1,       # "axis=1" (열)을 기준으로 삭제합니다.
#            inplace=True) # "inplace=True" 옵션을 사용하여, 결측치가 있는 열(여기서는 0번 열)을 
#                          # "원본 DataFrame df2에서 영구적으로 삭제"합니다.
# print(f"df2.dropna(axis=1,inplace=True) >>>> {df2}") # 이 코드를 실행한 후, df2는 0번 열이 사라진 상태가 됩니다.


# # --- 14. 결측치 삽입 ---
# df2.iloc[0,0] = np.nan # .iloc를 사용하여 "첫 번째 행(인덱스 0)", "첫 번째 열(인덱스 0)" 위치에 
#                        # "NaN (결측치)" 값을 할당합니다.


# # --- 15. 결측치 대체 (fillna) ---
# df2.fillna(0) # DataFrame 내의 모든 "NaN 값"을 "0"으로 대체한 "새로운 DataFrame"을 반환합니다.
#               # (이 결과를 변수에 할당하지 않았으므로, df2 원본에는 "변경 사항이 적용되지 않습니다".)
#               # 만약 원본을 변경하려면 `df2.fillna(1, inplace=True)`를 사용해야 합니다.

# df2.fillna(1) # DataFrame 내의 모든 "NaN 값"을 "1"로 대체한 "새로운 DataFrame"을 반환합니다.
#               # (위와 마찬가지로 원본 df2에는 "변경 사항이 적용되지 않습니다".)
#               # 만약 원본을 변경하려면 `df2.fillna(1, inplace=True)`를 사용해야 합니다.

# df2.fillna(1,inplace=True) # .fillna(1, inplace=True): DataFrame 내의 "모든 결측치(NaN)"를 "값 1"로 채웁니다.
# print(f"df2.fillna(1,inplace=True) >>>> {df2}") # inplace=True: 이 연산을 수행한 결과를 "원본 DataFrame(df2)"에 즉시 반영합니다.




# # --- 16. 결측치 삽입 ---
# df2.iloc[0,0] = np.nan # .iloc를 사용하여 "첫 번째 행(인덱스 0), 첫 번째 열(인덱스 0)" 위치에 "NaN (결측치)"을 할당합니다.

# # --- 17. 결측치 처리 및 타입 변환 (원본 변경 없음) ---
# # df2.fillna(0): DataFrame 내의 모든 "NaN (결측치)"를 "0"으로 대체합니다. (원본 df2는 변경되지 않음)
# # .astype(int): 결측치가 0으로 채워진 DataFrame의 모든 데이터 타입을 "정수(int)"로 변환합니다.
# df2.fillna(0).astype(int)

# # .astype(float): 결측치가 0으로 채워진 DataFrame의 모든 데이터 타입을 "실수(float)"로 변환합니다.
# df2.fillna(0).astype(float)

# # df2.fillna(0)[1]: 결측치가 0으로 채워진 DataFrame에서 "인덱스 1"에 해당하는 "열"을 Series 형태로 선택합니다.
# # .astype(int): 선택된 "1번 열"의 데이터 타입만을 "실수(float)"로 변환합니다.
# df2.fillna(0)[1].astype(float)

# print(f"df2.fillna(0)[1].astype(float) >>>> {df2}")


# # --- 18. 결측치 처리 및 타입 변환 ---
# # df2.fillna(0): DataFrame df2에서 존재하는 "모든 결측치(NaN)"를 "0"으로 채워 넣습니다.
# # .astype(float): 결측치를 채운 후, DataFrame의 모든 값의 데이터 타입을 "실수형(float)"으로 변환합니다.
# test_df = df2.fillna(0).astype(float) # 이 모든 처리가 완료된 결과를 test_df 변수에 저장합니다.

# print(test_df) # 최종 DataFrame test_df를 출력합니다.









# # --- DataFrame 생성 ---
# df3 = pd.DataFrame({ # pd.DataFrame() 함수를 사용하여 DataFrame을 생성하고 변수 df3에 저장합니다. 데이터는 Python 딕셔너리 형태로 제공됩니다.
#         'a':[1,3,4,3,4], # 딕셔너리의 키 'a'는 열(Column) 이름이 되며, 값 [1, 3, 4, 3, 4]가 해당 열의 데이터가 됩니다.
#         'b':[2,3,1,4,5], # 키 'b'는 열 이름, 값 [2, 3, 1, 4, 5]는 해당 열의 데이터입니다.
#         'c':[1,5,2,4,4]  # 키 'c'는 열 이름, 값 [1, 5, 2, 4, 4]는 해당 열의 데이터입니다.
# }) # 행 인덱스(index)가 별도로 지정되지 않았으므로, 0부터 시작하는 기본 정수 인덱스(0, 1, 2, 3, 4)가 사용됩니다.


# # --- 1. 리스트 요소 합산 ---
# np.sum([1,2,3]) # NumPy의 "sum 함수"를 호출하여 리스트 [1, 2, 3]의 모든 요소를 합산합니다.
#                 # 결과는 1 + 2 + 3 = 6이 됩니다.
#                 # 이 함수는 파이썬의 기본 함수 sum()과 유사하지만, NumPy 배열에 더 최적화되어 있습니다.
#                 # (이 결과는 출력되지 않고 메모리에만 생성됩니다.)
                

# # --- 2. 열별 합계 계산 ---
# # df3['a']: DataFrame df3에서 'a' 열(Column)을 "Series 형태"로 선택합니다.
# # np.sum(...): NumPy의 sum 함수를 호출하여 해당 Series (열)의 모든 값의 합계를 계산합니다.
# np.sum(df3['a']) # 'a' 열의 모든 값 (1 + 3 + 4 + 3 + 4)의 합계 15를 계산합니다.
# np.sum(df3['b']) # 'b' 열의 모든 값 (2 + 3 + 1 + 4 + 5)의 합계 15를 계산합니다.
# np.sum(df3['c']) # 'c' 열의 모든 값 (1 + 5 + 2 + 4 + 4)의 합계 16을 계산합니다.
# # 계산된 값들은 변수에 할당되지 않아 메모리에만 존재하며 출력되지는 않습니다.


# # --- 3. 반복문(for loop)을 이용한 열별 합계 계산 ---
# for i in ['a','b','c'] : # 열 이름('a', 'b', 'c') 리스트를 순회하는 반복문을 실행합니다.
#     # df3[i]: 현재 반복 중인 열 이름(i)에 해당하는 Series(열)를 선택합니다.
#     # np.sum(...): 선택된 Series의 모든 값의 합계를 계산합니다.
#     print(np.sum(df3[i])) # 각 열의 합계를 콘솔에 출력합니다.
#                           # (결과: 15, 15, 16)


# # --- 4. apply() 메서드를 이용한 열별 합계 계산 (Pandas 방식) ---
# # df3.apply(...): DataFrame의 각 열(또는 행)에 함수(여기서는 np.sum)를 적용하는 메서드입니다.
# # np.sum: 적용할 함수입니다.
# # 0: "axis=0"의 축(axis)을 의미합니다. "열(Column) 방향", 즉 "행(Row) 간의 연산"을 수행하라는 뜻입니다.
# #    (각 열의 모든 행에 대해 np.sum을 적용하므로 열별 합계를 계산합니다.)
# df3.apply(np.sum, 0)  # 이 코드는 열별 합계를 계산한 Series를 반환합니다.
#                       # (변수에 할당되거나 print 되지 않았으므로 출력은 되지 않습니다.)
#                       # (결과: Series, Index: a, b, c, Value: 15, 15, 16)


# # --- 5. 사용자 정의 함수 정의 ---
# def diff(x) : # 'diff'라는 이름의 함수를 정의합니다. 이 함수는 하나의 인자(x)를 받습니다.
#     return x.max() - x.min() # 함수는 x의 최댓값(x.max())에서 최솟값(x.min())을 뺀 값, 즉 "최대-최소 차이(범위)"를 반환합니다.
# # 주석: 이 함수는 나중에 df3.apply(diff, axis=...)와 같이 사용될 때,
# #       인자 x는 df3의 각 "열(Column) 또는 행(Row) Series"가 됩니다.
# #       (예: 'a' 열 Series가 x로 전달되면, 4 - 1 = 3이 반환됩니다.)


# # --- 6. 사용자 정의 함수 사용 ---
# print(diff(df3['a'])) # DataFrame df3에서 "'a' 열(Series)"을 선택하여 diff 함수에 전달하고 결과를 계산합니다.
#                       # 'a' 열의 최댓값(4) - 최솟값(1) = 3이 계산됩니다.
# print(diff(df3['b'])) # DataFrame df3에서 "'b' 열(Series)"을 선택하여 diff 함수에 전달하고 결과를 계산합니다.
#                       # 'b' 열의 최댓값(5) - 최솟값(1) = 4가 계산됩니다.
# print(diff(df3['c'])) # DataFrame df3에서 "'c' 열(Series)"을 선택하여 diff 함수에 전달하고 결과를 계산합니다.
#                       # 'c' 열의 최댓값(5) - 최솟값(1) = 4가 계산됩니다.
#                       # (이 세 호출 결과는 변수에 할당되거나 print되지 않아 화면에는 출력되지 않습니다.)
               

# # --- 7. 함수 적용 및 출력 ---
# print(df3.apply(diff, 0)) # DataFrame의 ".apply() 메서드"를 사용하여 정의한 'diff' 함수를 적용합니다.
#                           # diff: DataFrame의 각 열(Series)에 적용될 함수입니다.
#                           # 0: "axis=0"을 의미하며, "열(Column) 방향"으로 함수를 적용하라는 뜻입니다.
#                           #    결과적으로 각 열 'a', 'b', 'c'에 대해 "독립적인 최대-최소 차이"가 계산됩니다.
#                           # 계산된 결과(Series)를 콘솔에 출력합니다.


# # --- 8. 람다 함수(Lambda Function) 사용 ---
# print((lambda x : x.max()-x.min())(df3['a'])) # "람다 함수(익명 함수)"를 정의하고 즉시 실행합니다.
#                                               # lambda x : x.max()-x.min()은 `def diff(x): return x.max() - x.min()`과 동일한 역할을 합니다. 
#                                               # 위의 5번 함수와 같은 기능임.
#                                               # df3['a'] 열(Series)을 인수로 전달하여 실행합니다.
#                                               # 'a' 열의 최대-최소 차이(3)를 계산하고 반환합니다.
#                                               # (print 명령이 없으므로 이 결과는 화면에 출력되지 않습니다.)


# # --- 9. apply()를 이용한 전체 열에 함수 적용 (람다 함수) ---
# print(df3.apply(lambda x : x.max()-x.min(),0)) # 람다 함수(lambda function)를 사용하여 "동일한 최대-최소 차이 연산"을 수행합니다.
#                                                # lambda x: x.max()-x.min()은 diff 함수와 동일한 기능을 즉석에서 정의합니다.
#                                                # 0: "axis=0" (열 방향)으로 적용하여 각 열별 최대-최소 차이를 계산합니다.
                                        

# # --- 10. Pandas 내장 메서드를 이용한 효율적인 계산 ---
# # df3.max(axis=0): 각 "열(Column)별 최댓값"을 Series로 계산합니다. (결과: a=4, b=5, c=5)
# # df3.min(axis=0): 각 "열(Column)별 최솟값"을 Series로 계산합니다. (결과: a=1, b=1, c=1)
# # 두 Series 간에 뺄셈 연산을 수행하여 각 열의 "최대-최소 차이(범위)"를 계산합니다.
# # 이 방법은 사용자 정의 함수나 .apply()보다 pandas 내부적으로 더 최적화되어 있어 "가장 효율적인 방법"입니다.
# print(df3.max(axis=0) - df3.min(axis=0))


# # --- 11. NumPy 함수 사용 (개별 호출) ---
# np.square([1,2,3]) # NumPy의 "np.square()" 함수를 사용하여 배열 [1, 2, 3]의 각 원소를 제곱합니다.
#                    # 결과 ([1, 4, 9])는 반환되지만, 변수에 할당되거나 print 되지 않아 출력되지 않습니다.


# # --- 12. 원본 DataFrame 출력 ---
# print(df3) # 원본 DataFrame df3를 출력합니다.


# # --- 13. apply()를 이용한 열별 NumPy 함수 적용 ---
# # df3.apply(np.square, 0): np.square 함수를 "열(Column) 방향(axis=0)"으로 적용합니다.
# #                         각 열의 모든 원소를 독립적으로 제곱하여 새로운 DataFrame을 반환합니다.
# print(df3.apply(np.square,0)) # 제곱된 결과 DataFrame을 출력합니다.


# # --- 14. apply()를 이용한 행별 NumPy 함수 적용 ---
# # df3.apply(np.square, 1): np.square 함수를 "행(Row) 방향(axis=1)"으로 적용합니다.
# #                         각 행의 모든 원소를 독립적으로 제곱하여 새로운 DataFrame을 반환합니다.
# print(df3.apply(np.square,1)) # 제곱된 결과 DataFrame을 출력합니다.













# # seaborn 라이브러리가 제공하는 "'titanic' 데이터셋"을 불러옵니다.
# # 이 함수는 데이터를 pandas "DataFrame" 형태로 반환하며, 
# # 그 결과를 'titanic' 변수에 할당합니다.
# # 'titanic' 데이터셋은 타이타닉호 승객에 대한 정보(나이, 성별, 생존 여부 등)를 포함하고 있습니다.
# titanic = sns.load_dataset('titanic') 


# # --- 1. 특정 열 선택 및 빈도수 계산 ---
# # titanic[['alive','sex','class']]: 원본 DataFrame에서 'alive', 'sex', 'class' 세 열만 선택하여 새로운 DataFrame을 생성합니다.
# # .apply(pd.value_counts, 0): 선택된 DataFrame에 ".apply() 메서드"를 사용하여 함수를 적용합니다.
# #   - pd.value_counts: DataFrame의 각 열(Series)에 적용될 함수입니다. 각 열의 고유한 값별 빈도수를 계산합니다.
# #   - 0: "axis=0"을 의미합니다. "열(Column) 방향"으로 함수를 적용하라는 뜻입니다.
# #        즉, 'alive' 열의 빈도, 'sex' 열의 빈도, 'class' 열의 빈도를 각각 독립적으로 계산합니다.

# print(titanic[['alive','sex','class']].apply(pd.value_counts,0)) # 계산된 결과를 콘솔에 출력합니다.
#                                                                   # 결과는 각 열의 빈도수가 합쳐진 DataFrame 형태로 출력됩니다.
#                                                                   # 빈도수가 없는 항목은 NaN으로 표시될 수 있습니다.
                                                                  







# # --- 1. Series 생성 ---
# # pd.Series(3): 값으로 "단일 값(3)"을 가지는 Series를 생성합니다.
# #               인덱스가 지정되지 않았으므로 0부터 시작하는 기본 정수 인덱스(0)가 사용됩니다.
# print(pd.Series(3)) # 생성된 Series를 출력합니다.
#                     # 결과: 
#                     # 0    3
#                     # dtype: int64

# # --- 2. 최댓값 계산 및 출력 ---
# # pd.Series(3).max(): 단일 값(3)을 가진 Series에 대해 ".max() 메서드"를 호출합니다.
# #                     Series에 값이 하나밖에 없으므로, 그 값(3)이 최댓값이 됩니다.
# print(pd.Series(3).max()) # Series의 최댓값을 계산하여 출력합니다.
#                           # 결과: 3
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
# # --- 1. 데이터 및 기준 정의 ---
# ages=[0,0.5,4,6,4,5,2,10,21,23,37,15,38,31,61,20,41,31,100] # 분석할 나이(ages) 데이터 리스트를 정의합니다.
# data = ages # ages 리스트를 data 변수에 할당합니다.
# bins = [0,4,18,25,35,60,100] # 데이터를 나눌 구간(경계 값) 리스트를 정의합니다. (총 6개의 구간이 생성됩니다.)
#                              # 구간: (0~4), (4~18), (18~25), (25~35), (35~60), (60~100)
# labels = ['영유아','미성년자','청년','중년','장년','노년'] # 각 구간에 할당할 레이블(범주 이름) 리스트를 정의합니다.
#                                                         # 레이블 개수(6개)는 구간의 개수와 일치해야 합니다.

# # --- 2. 데이터 크기 확인 ---
# print(len(data)) # data 리스트의 길이(총 원소 개수)를 출력합니다. (결과: 19)


# # --- 3. 데이터 범주화 (Categorization) ---
# # pd.cut(): "연속적인 수치 데이터"를 지정된 구간(bins)에 따라 "범주형 데이터"로 나눕니다.
# cats = pd.cut(data,       # 범주화할 데이터 (나이 리스트)
#               bins,       # 나이 구간 경계 리스트
#               labels=labels) # 각 구간에 붙일 레이블 리스트

# print(cats) # 범주화된 결과(cats)를 출력합니다.
#             # 결과는 pandas의 "Categorical" 타입 객체입니다.
            
# print(type(cats)) # cats 객체의 타입을 출력합니다. (결과: <class 'pandas.core.arrays.categorical.Categorical'>)


# # --- 4. 결과 리스트 변환 및 출력 ---
# cat_list = list(cats) # Categorical 객체인 cats를 Python의 일반 리스트로 변환합니다.
# print(cat_list) # 리스트로 변환된 범주화 결과(각 나이에 해당하는 범주 이름)를 출력합니다.


# # --- 5. DataFrame 생성 및 범주 추가 ---
# # DataFrame 생성: 나이 데이터(ages)와 범주 리스트(cat_list)를 사용하여 '나이'와 '연령대' 두 열을 갖는 DataFrame을 생성합니다.
# test = pd.DataFrame({'나이':ages,'연령대':cat_list})
# print(test) # 생성된 DataFrame을 출력합니다.

# # --- 6. 범주별 빈도수 계산 ---
# print(test['연령대'].value_counts()) # DataFrame의 '연령대' 열(Series)을 선택한 후,
#                                     # `.value_counts()` 메서드를 사용하여 각 범주별로 데이터가 몇 개씩 있는지 (빈도수)를 계산합니다.
#                                     # 이 결과는 Series 형태로 반환됩니다.
#                                     # (변수에 할당되거나 print 되지 않아 최종적으로 출력되는 결과는 value_counts()의 결과입니다.)


# # --- 7. 타입 확인 (반복) ---
# print(type(cats)) # cats 객체의 타입을 다시 출력합니다
#                   # <class 'pandas.core.arrays.categorical.Categorical'>

# print(cats.categories) # Categorical 객체가 가지고 있는 "범주(Category)" 레이블들을 출력합니다.
#                        # (결과: Index(['영유아', '미성년자', '청년', '중년', '장년', '노년'], dtype='object'))

# print(cats.codes)      # Categorical 객체가 가지고 있는 "각 데이터 포인트가 속한 범주의 정수 코드"를 출력합니다.
#                        # (결과: [-1, 0, 0, 1, 0, 1, 0, 1, 2, 2, 4, 1, 4, 3, 5, 2, 4, 3, 5])
#                        # 코드 -1은 해당 값이 어느 구간에도 속하지 않음(경계 밖)을 의미합니다.












# # --- 1. 시드 설정 및 데이터 생성 ---
# np.random.seed(2) # 난수 생성기의 "시드(Seed)"를 "2"로 고정합니다 (결과 재현성 보장).

# # np.random.randint(20, size=20): 0 (포함)부터 20 (미포함) 사이의 정수 난수 20개를 생성합니다.
# data = np.random.randint(20, size = 20) # 이 난수들을 1차원 NumPy 배열 'data'에 저장합니다.

# print(data) # 생성된 원본 데이터 배열을 출력합니다.
#             # (결과: [ 8 13  2 11  2  4 10 13 14  6  1 18  6  8  5 14 16 11 15 11])


# # --- 2. 사분위수 기반 그룹화 (Quantile Cut) ---
# qcat = pd.qcut(data,     # 그룹화할 원본 데이터 (20개의 랜덤 정수)
#                4,        # 데이터를 나눌 그룹의 개수 (4개, 즉 사분위수)를 지정합니다. 
#                          # 데이터를 4등분하여 Q1, Q2, Q3, Q4 네 구간으로 나눕니다.
#                labels=['Q1','Q2','Q3','Q4']) # 각 4개의 그룹에 할당할 범주 이름(레이블)을 지정합니다.
#                                              # qcut은 데이터의 "개수"를 기준으로 동일하게 나누려 합니다.

# print(qcat) # 그룹화된 결과 Series를 출력합니다.
#             # 이 결과는 pandas의 "Categorical" 타입입니다.
#             # (각 데이터 포인트가 Q1~Q4 중 어느 그룹에 속하는지 표시됩니다.)


# # --- 3. 데이터 정렬 및 확인 ---
# print(np.sort(data)) # 원본 배열 'data'를 "오름차순"으로 정렬하여 출력합니다.
#                      # 이 정렬된 배열을 통해 사분위수 경계가 대략 어디에 위치하는지 유추할 수 있습니다.


# # --- 4. 빈도 분석 및 출력 ---
# print(qcat.value_counts()) # Categorical 객체 'qcat'에 대해 `.value_counts()`를 호출하여
#                            # "각 분위수 범주별 빈도수(개수)"를 계산하고 출력합니다.
#                            # qcut은 데이터의 개수를 동일하게 나누려고 시도하므로, 
#                            # 20개 데이터를 4개 분위수로 나누면 이상적으로는 각 분위수가 5개씩 나와야 합니다.
#                            # (결과: Q1, Q2, Q3, Q4 모두 5개가 나옴)


# # --- 5. DataFrame 생성 및 열 추가 ---
# df0 = pd.DataFrame(data, # 원본 데이터 'data'를 사용하여 DataFrame을 생성합니다.
#                    columns=['관측수']) # 열 이름을 '관측수'로 지정합니다.
# print(df0) # 1열 DataFrame df0를 출력합니다.

# df0['범주'] = qcat # DataFrame df0에 "'범주'"라는 이름의 "새로운 열"을 추가하고,
#                    # 이전에 계산된 분위수 범주 데이터인 "qcat Series의 값들"을 할당합니다.
# print(df0) # '관측수'와 '범주' 두 열을 갖게 된 최종 DataFrame df0를 출력합니다.














# # --- 1. DataFrame 생성 ---
# df3 = pd.DataFrame({ # DataFrame을 생성하여 변수 df3에 저장합니다.
#     'a':[1,3,4,3,4], # 'a' 열 데이터
#     'b':[2,3,1,4,5], # 'b' 열 데이터
#     'c':[1,5,2,4,4]  # 'c' 열 데이터
# })
# # 초기 df3의 인덱스는 0부터 4까지의 기본 정수입니다.
# # 초기 df3:
# #    a  b  c
# # 0  1  2  1
# # 1  3  3  5
# # 2  4  1  2
# # 3  3  4  4
# # 4  4  5  4


# # --- 2. 인덱스 설정 시도 (원본 변경 안됨) ---
# df3.set_index('a') # 'a' 열의 값을 새로운 "행 인덱스"로 설정합니다.
#                    # 이 명령은 인덱스가 변경된 "새로운 DataFrame"을 반환하지만,
#                    # 그 결과를 변수에 저장하지 않았고, `inplace=True` 옵션이 없으므로
#                    # "원본 DataFrame df3는 변경되지 않고" 그대로 유지됩니다.

# print(df3)         # "원본 df3"를 출력합니다. (여전히 'a' 열은 데이터 열로 남아있고, 인덱스는 0, 1, 2, 3, 4입니다.)
#                    # 결과: (초기 df3와 동일)


# # --- 3. 인덱스 설정 (원본 변경됨) ---
# df3.set_index('a', # 'a' 열의 값을 새로운 "행 인덱스"로 설정합니다.
#               inplace=True) # "inplace=True" 옵션을 사용하여 "원본 DataFrame df3 자체"를 변경합니다.
#                             # 이제 'a' 열은 더 이상 데이터 열이 아니며, 인덱스(레이블) 역할을 수행합니다.
#                             # 'a' 열은 중복된 값(3, 4)을 포함하므로, 인덱스에도 중복이 발생합니다.

# print(df3)         # "변경 완료된 df3"를 출력합니다. ('a' 열이 인덱스가 되었고, 열 개수는 2개('b', 'c')로 줄었습니다.)
# # 변경된 df3:
# #    b  c
# # a      
# # 1  2  1
# # 3  3  5
# # 4  1  2
# # 3  4  4
# # 4  5  4


# # --- 4. 인덱스 레이블로 행 조회 ---
# print(df3.loc[3])  # "`.loc`" 접근자를 사용하여 "인덱스 레이블이 '3'인 모든 행"을 조회합니다.
#                    # 'a' 열에 값이 '3'인 행이 두 개 있었으므로, 두 행이 모두 반환됩니다.
#                    # (인덱스 3에 해당하는 모든 데이터가 포함된 DataFrame 또는 Series가 출력됩니다.)


# # --- 5. 인덱스 재설정 시도 (원본 변경 없음) ---
# # 현재 인덱스(이전에 'a' 열이었던 값)를 데이터 열로 되돌리고, 
# # 0부터 시작하는 새로운 기본 정수 인덱스를 설정합니다.
# # 하지만 결과를 변수에 할당하지 않고 `inplace=False` (기본값)이므로,
# # "원본 DataFrame df3는 변경되지 않습니다." (여전히 인덱스는 [1, 3, 4, 3, 4]입니다.)
# df3.reset_index() 
# print(df3) # "재설정 시도 후 df3"를 출력합니다. (인덱스는 여전히 [1, 3, 4, 3, 4]입니다. 3번 출력 결과와 같습니다.)


# # --- 6. 인덱스 재설정 + 인덱스 열 제거 (원본 미변경) ---
# df3.reset_index(drop=True) # 인덱스 재설정을 수행하되,
#                            # "drop=True"를 사용하여 "현재 인덱스('a')를 일반 열로 되돌리지 않고" 그냥 버립니다.
#                            # 새로운 0부터 시작하는 기본 정수 인덱스가 설정된 "새로운 DataFrame"을 반환합니다.
#                            # (원본 df3는 변경되지 않습니다. 인덱스는 여전히 'a'입니다.)
# print(df3)                 # 원본 df3를 출력합니다. (인덱스는 여전히 'a'입니다.)


# # --- 7. 인덱스 재설정 + 인덱스 열 제거 (원본 변경) ---
# df3.reset_index(drop=True, # 현재 인덱스('a')를 버리고,
#                 inplace=True) # "inplace=True"를 사용하여 "원본 df3를 영구적으로 변경"합니다.
#                               # df3는 이제 0부터 시작하는 기본 정수 인덱스만 갖게 됩니다.
# print(df3)                    # 최종 변경된 df3를 출력합니다. (인덱스: 0, 1, 2, 3, 4 / 열: b, c)



# # --- 8. 인덱스 및 컬럼 이름 변경 (rename) ---
# df3.rename(index={0:'1반',1:'2반'}) # 인덱스 레이블 0을 '1반'으로, 1을 '2반'으로 이름을 변경합니다.
#                                     # "inplace=False"이므로, "원본 df3는 변경되지 않습니다".

# df3.rename(columns={'b':'학생'},inplace=True) # 컬럼 이름 'b'를 '학생'으로 변경합니다.
#                                             # "inplace=True"를 사용하여 "원본 df3 자체를 변경"합니다.

# print(df3) # 원본 df3를 출력합니다. (컬럼 'b'가 '학생'으로 변경되어 있습니다. 인덱스는 그대로 정수 0~4입니다.)













# # --- 1. 리스트 초기화 ---
# a = [1, 2, 3, 4] # 정수 1, 2, 3, 4를 요소로 가지는 Python 리스트 'a'를 생성하고 초기화합니다.
# result = []      # 연산 결과를 저장할 빈 리스트 'result'를 생성합니다.

# # --- 2. 반복문(for loop)을 이용한 연산 및 저장 ---
# for num in a : # 리스트 'a'의 각 요소를 순서대로 하나씩 꺼내어 임시 변수 'num'에 할당하면서 반복합니다.
#     # 첫 번째 반복: num = 1
#     # 두 번째 반복: num = 2
#     # ...

#     result.append(num * 2) # 현재 요소 'num'에 2를 곱한 결과를 계산하고,
#                            # 그 결과를 'result' 리스트의 "가장 마지막"에 추가합니다.
#                            # 1 * 2 = 2가 result에 추가
#                            # 2 * 2 = 4가 result에 추가
#                            # 3 * 2 = 6이 result에 추가
#                            # 4 * 2 = 8이 result에 추가

# # --- 3. 결과 출력 ---
# print(result) # 최종 연산 결과가 저장된 리스트 'result'를 출력합니다.
#               # (결과: [2, 4, 6, 8])


# # --- 4. 리스트 컴프리헨션(List Comprehension)을 사용한 리스트 생성 ---
# # 리스트 컴프리헨션은 반복문과 동일한 작업을 한 줄로 간결하고 효율적으로 수행하는 Python의 강력한 기능입니다.
# # 문법: [ 표현식(Expression) for 항목(Item) in 반복 가능한 객체(Iterable) ]
# result = [num * 2 for num in a]
# # 해석: 리스트 'a'의 각 'num'에 대해 'num * 2'를 계산하여 새로운 리스트 'result'에 저장합니다.

# print(result) # 새로 생성된 'result' 리스트를 출력합니다. (결과: [2, 4, 6, 8])










# # --- 1. 초기화 ---
# test = [] # 결과를 저장할 빈 리스트 'test'를 초기화합니다.
#           # 이 리스트에 'id_1', 'id_2' 등의 문자열이 순서대로 저장될 것입니다.

# # --- 2. 반복문(for loop)을 사용한 리스트 생성 ---
# for i in range(1,5) : # range(1, 5)는 1부터 시작하여 5 미만인 정수(1, 2, 3, 4)를 순서대로 생성합니다.
#     # 'i' 변수는 각 반복마다 1, 2, 3, 4의 값을 가집니다.
    
#     # 'id_' + str(i):
#     #   - 문자열 'id_'와 현재 정수 값 'i'를 문자열로 변환한 값(str(i))을 결합(concatenation)합니다.
#     #   - 예: i=1일 때 'id_1', i=4일 때 'id_4'가 생성됩니다.
#     test.append('id_'+str(i)) # 생성된 문자열 식별자를 리스트 'test'의 끝에 추가합니다.

# print(test) # 최종적으로 완성된 리스트 'test'를 출력합니다.
#             # (결과: ['id_1', 'id_2', 'id_3', 'id_4'])
            
            
# # --- 3. 리스트 컴프리헨션(List Comprehension)을 사용한 리스트 생성 ---
# # 이 방법은 동일한 작업을 한 줄로 간결하게 수행합니다.
# # 문법: [ 표현식 for 항목 in 반복 가능한 객체 ]
# test = ['id_'+str(i) for i in range(1,5) ] 
# # 해석: range(1, 5)의 각 'i'에 대해 ('id_' + str(i))를 계산하여 새로운 리스트 'test'를 생성합니다.

# print(test) # 새로 생성된 'test' 리스트를 출력합니다. (결과: ['id_1', 'id_2', 'id_3', 'id_4'])