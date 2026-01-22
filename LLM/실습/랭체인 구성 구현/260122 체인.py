# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os  # 운영체제 상호작용 (환경변수 접근 등)
from dotenv import load_dotenv  # .env 파일에서 환경변수 로드
from langchain_classic.chains import LLMChain  # LLM과 프롬프트를 연결하는 기본 체인 클래스
from langchain_classic import PromptTemplate  # 프롬프트 템플릿 생성을 위한 클래스
from langchain_classic.chat_models import ChatOpenAI  # OpenAI 채팅 모델 연동 클래스
from langchain_classic.chains import SequentialChain  # 여러 입력/출력을 지원하는 순차 체인
from langchain_classic.chains import SimpleSequentialChain  # 단일 입력/출력을 지원하는 간단한 순차 체인

# =============================================================================
# [2] 환경 변수 로드 (Load Environment Variables)
# =============================================================================
# .env 파일에 저장된 OPENAI_API_KEY 등을 시스템 환경변수로 로드합니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# [3] LLM 초기화 (Initialize LLM)
# =============================================================================
# ChatOpenAI 객체 생성
# temperature=0: 답변의 창의성(무작위성)을 0으로 설정하여 일관된 답변을 유도합니다.
# model_name: 사용할 모델명을 지정합니다. (gpt-4o-mini 등 표준 모델명 사용 권장)
llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')

# =============================================================================
# [4] 기본 LLMChain 테스트 (Basic LLMChain Usage)
# =============================================================================
# 1. 프롬프트 템플릿 정의: {country} 변수가 포함된 질문 양식을 만듭니다.
prompt = PromptTemplate(
    input_variables=["country"],
    template="{country}의 수도는 어디야?",
)

# 2. LLMChain 생성: 모델(llm)과 질문 양식(prompt)을 하나로 결합합니다.
chain = LLMChain(llm=llm, prompt=prompt)

# 3. 실행: 대한민국을 입력값으로 주어 실행 결과를 출력합니다.
print("\n--- [4] 단일 LLMChain 실행 결과 ---")
print(chain.run("대한민국"))

# =============================================================================
# [5] 순차 체인 구성을 위한 개별 체인 정의 (Individual Chains for Pipeline)
# =============================================================================

# --- 체인 1: 번역 (Translation) ---
# 입력된 문장을 한국어로 번역하는 역할을 수행합니다.
prompt1 = PromptTemplate(
    input_variables=['sentence'],
    template="다음 문장을 한글로 번역하세요.\n\n{sentence}"
)
# output_key: 이 체인의 출력값을 "translation"이라는 키값으로 저장하여 다음 단계에서 쓸 수 있게 합니다.
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="translation")

# --- 체인 2: 요약 (Summary) ---
# 번역된 문장을 한 문장으로 요약하는 역할을 수행합니다.
prompt2 = PromptTemplate.from_template(
    "다음 문장을 한 문장으로 요약하세요.\n\n{translation}"
)
# output_key: 출력값을 "summary"라는 키값으로 저장합니다.
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")

# =============================================================================
# [6] SimpleSequentialChain 사용 (SimpleSequentialChain Usage)
# =============================================================================
# 가장 단순한 형태의 순차 체인으로, 오직 "하나의 입력"과 "하나의 출력"만 허용합니다.
# 중간 단계의 데이터(번역본 등)는 최종 결과에 포함되지 않고 마지막 응답만 반환합니다.
overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2], # 실행할 체인들을 순서대로 리스트에 담습니다.
    verbose=True             # 실행되는 중간 과정을 터미널에 상세히 보여줍니다.
)

print("\n--- [6] SimpleSequentialChain 시작 ---")
overall_chain.run("The quick brown fox jumps over the lazy dog.")

# =============================================================================
# [7] SequentialChain 사용 (SequentialChain Usage)
# =============================================================================
# 여러 개의 입력값이나 중간 단계의 출력값(translation)을 모두 결과에 포함하고 싶을 때 사용합니다.
all_chain = SequentialChain(
    chains=[chain1, chain2],            # 구성할 체인 리스트
    input_variables=['sentence'],       # 전체 체인의 시작 입력 변수
    output_variables=['translation','summary'], # 최종 결과 딕셔너리에 포함될 출력 변수들
)

# 테스트용 긴 영문 텍스트 (LangChain 로더 관련 설명)
sentence = """
One limitation of LLMs is their lack of contextual information
(e.g., access to some specific documents or emails).
You can combat this by giving LLMs access to the specific external data.
For this, you first need to load the external data with a document loader.
LangChain provides a variety of loaders for different types of documents ranging
from PDFs and emails to websites and YouTube videos.
"""

print("\n--- [7] SequentialChain 실행 결과 ---")
# 딕셔너리 형태로 번역 결과와 요약 결과가 모두 포함된 데이터를 반환합니다.
result = all_chain(sentence)
print(result)