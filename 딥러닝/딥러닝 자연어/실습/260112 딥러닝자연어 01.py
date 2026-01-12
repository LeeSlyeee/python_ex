# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# Keras의 텍스트 전처리 유틸리티 임포트
from tensorflow.keras.preprocessing.text import Tokenizer            # 텍스트를 토큰화하기 위한 클래스
from tensorflow.keras.preprocessing.sequence import pad_sequences    # 시퀀스 데이터의 길이를 맞추기 위한 함수

# Keras 모델 및 레이어 관련 임포트
from tensorflow.keras.models import Sequential                       # 순차적 모델 구성을 위한 클래스
from tensorflow.keras.layers import Dense,Flatten,Embedding          # 완전 연결 층, 평탄화 층, 임베딩 층
from tensorflow.keras.utils import to_categorical                    # 원-핫 인코딩 유틸리티
from numpy import array                                              # 배열 통일성을 위해 numpy의 array 임포트


from tensorflow.keras.preprocessing.text import text_to_word_sequence # 단어 시퀀스 생성 함수 직접 임포트


# =============================================================================
# [2] 텍스트 토큰화 기초 (Text Tokenization Basics)
# =============================================================================
# 예제 텍스트 정의
text = '해보지 않으면 해낼 수 없다'

# text_to_word_sequence: 문장을 단어(토큰) 단위로 분리 (기본적으로 공백 기준)
result = text_to_word_sequence(text)

# 원문 및 결과 출력
print("\n원문:\n", text)
print("\n토큰화:\n", result)



# =============================================================================
# [3] Tokenizer 객체 활용 및 단어 빈도 분석 (Tokenizer Usage & Frequency Analysis)
# =============================================================================
# 분석할 문서(문장) 리스트 정의
docs = [
  '먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
  '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
  '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.'
]


# Tokenizer 객체 생성
token = Tokenizer()
# fit_on_texts: 문서 리스트를 입력받아 내부 단어 집합(Vocabulary) 생성 및 빈도수 학습
token.fit_on_texts(docs)

# word_counts: 단어별 등장 횟수 (OrderedDict 형태)
print("\n단어 카운트:\n", token.word_counts)


# document_count: 학습된 문서(문장)의 총 개수
print("\n문장 카운트: ", token.document_count)

# word_docs: 각 단어가 몇 개의 문장에 등장했는지 카운트
print("\n각 단어가 몇 개의 문장에 포함되어 있는가:\n", token.word_docs)

# word_index: 각 단어에 부여된 고유 인덱스 (빈도수 순으로 1부터 부여)
print("\n각 단어에 매겨진 인덱스 값:\n", token.word_index)


# =============================================================================
# [4] 단일 문장 토큰화 및 시퀀스 변환 (Sequence Conversion)
# =============================================================================
# 새로운 텍스트 데이터 정의
text="오랫동안 꿈꾸는 이는 그 꿈을 닮아간다"

# Tokenizer 객체 초기화 및 단일 텍스트 학습
token = Tokenizer()
token.fit_on_texts([text]) # 리스트 형태로 전달해야 함

# 생성된 단어 인덱스 출력
print(token.word_index)


# texts_to_sequences: 텍스트를 정수 인덱스 시퀀스로 변환
x=token.texts_to_sequences([text])
print(x) # 변환된 정수 시퀀스 출력



# =============================================================================
# [5] 원-핫 인코딩 (One-Hot Encoding)
# =============================================================================
# 단어 집합의 크기 계산 (인덱스가 1부터 시작하므로 +1)
word_size = len(token.word_index) + 1

# to_categorical: 정수 시퀀스를 원-핫 벡터로 변환
x = to_categorical(x, num_classes=word_size)
print(x) # 원-핫 인코딩 결과 출력



# =============================================================================
# [6] 텍스트 분류 실습 - 리뷰 긍정/부정 예측 (Review Sentiment Analysis)
# =============================================================================
# 리뷰 텍스트 데이터 (문서)
docs = [
  "너무 재밌네요",               # 긍정
  "최고예요",                   # 긍정
  "참 잘 만든 영화예요",           # 긍정
  "추천하고 싶은 영화입니다",        # 긍정
  "한번 더 보고싶네요",            # 긍정
  "글쎄요",                     # 부정
  "별로예요",                   # 부정
  "생각보다 지루하네요",           # 부정
  "연기가 어색해요",              # 부정
  "재미없어요"                  # 부정
]


# 각 리뷰에 대한 레이블(클래스) 정의: 긍정(1), 부정(0)
classes = array([1,1,1,1,1,0,0,0,0,0])


# Tokenizer 객체 생성
token = Tokenizer()
# 전체 문서에 대해 단어 인덱스 학습
token.fit_on_texts(docs)
print(token.word_index) # 단어 인덱스 출력

# 텍스트 데이터를 정수 시퀀스로 변환
x = token.texts_to_sequences(docs)
print("\n리뷰 텍스트, 토큰화 결과:\n", x)

# 패딩(Padding): 시퀀스 길이를 4로 고정 (부족하면 0으로 채움)
padded_x = pad_sequences(x, 4)
print("\n패딩 결과:\n", padded_x)

# 임베딩 입력을 위한 전체 단어 개수 파악 (+1은 패딩 인덱스 0 포함)
word_size = len(token.word_index) +1

# 모델 구축 (Sequential API)
model = Sequential()

# Embedding 층: 단어를 밀집 벡터(Dense Vector)로 변환
# word_size: 입력 단어 집합의 크기
# 8: 임베딩 벡터의 차원 (각 단어를 8개의 숫자로 표현)
# 입력 시퀀스 길이(input_length)는 pad_sequences의 4와 대응됨
model.add(Embedding(word_size, 8))

# Flatten 층: 2차원(단어 수 x 임베딩 차원) 출력을 1차원으로 평탄화
model.add(Flatten())

# Dense 층 (출력층): 이진 분류(0 또는 1)이므로 출력 노드는 1개, 활성화 함수는 시그모이드(sigmoid)
model.add(Dense(1, activation='sigmoid'))

# 모델 입력 형태 명시하며 빌드 (배치 크기: None, 시퀀스 길이: 4)
model.build(input_shape=(None,4))
model.summary() # 모델 구조 요약정보 출력

# 모델 컴파일
# optimizer='adam': 아담 최적화 알고리즘
# loss='binary_crossentropy': 이진 분류용 손실 함수
# metrics=['accuracy']: 정확도 측정
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
# padded_x: 입력 데이터, classes: 정답 레이블, epochs=20: 20회 반복 학습
model.fit(padded_x, classes, epochs=20)

# 학습된 모델 평가 및 정확도 출력
print("\n Accuracy: %.4f" % (model.evaluate(padded_x, classes)[1]))
