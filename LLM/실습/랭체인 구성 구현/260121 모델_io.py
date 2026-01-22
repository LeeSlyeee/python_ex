# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os  # 운영체제와 상호작용하기 위한 모듈 (환경변수 접근 등)
from dotenv import load_dotenv  # .env 파일에서 환경변수를 읽어오기 위한 라이브러리
from langchain_classic import PromptTemplate  # 프롬프트 템플릿을 생성하고 관리하는 클래스
from langchain_classic.chat_models import ChatOpenAI  # OpenAI의 채팅 모델(GPT 등)을 사용하기 위한 클래스
from langchain_classic.model_laboratory import ModelLaboratory  # 여러 LLM 모델의 출력을 비교할 수 있는 실험실 도구
from langchain_classic.output_parsers import CommaSeparatedListOutputParser  # LLM의 출력을 쉼표로 구분된 리스트로 파싱하는 도구

# =============================================================================
# [2] 환경 변수 로드 (Load Environment Variables)
# =============================================================================
# .env 파일에 저장된 API 키 값을 환경변수로 로드합니다.
load_dotenv()

# 시스템 환경변수에서 OpenAI 및 HuggingFace API 토큰을 가져옵니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# =============================================================================
# [3] PromptTemplate 기초 사용법 (Basic PromptTemplate Usage)
# =============================================================================
# 템플릿 문자열 정의: {product} 부분이 나중에 실제 값으로 대체됩니다.
template = "{product}를 홍보하기 위한 좋은 문구를 추천해줘?"

# PromptTemplate 객체 생성
# input_variables: 템플릿 안에서 변수처럼 바뀔 부분의 이름을 리스트로 지정
# template: 실제 템플릿 문자열
prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

# format 메서드를 사용하여 {product} 자리에 "카메라"를 넣어 완성된 문장을 생성하고 출력합니다.
print("\n--- [3] PromptTemplate 결과 ---")
print(prompt.format(product="카메라"))


# =============================================================================
# [4] ChatOpenAI 모델 초기화 및 기본 사용 (ChatOpenAI Initialization)
# =============================================================================
# 첫 번째 LLM 객체 생성 (OpenAI GPT 모델)
# temperature=0: 창의성(무작위성)을 0으로 설정하여 항상 일관된 답변을 하도록 유도
# model='gpt-4.1-mini': 사용할 모델명 지정 (실제 존재하는 모델명인지 확인 필요, 보통 gpt-4o-mini 등 사용)
llm1 = ChatOpenAI(temperature=0, model='gpt-4o-mini')  # 코드의 모델명을 표준 모델명으로 가정하고 주석 작성

# 모델에게 질문할 프롬프트 질문 내용 정의
prompt_text = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"

# llm1.invoke(): 모델에게 질문을 보내고 응답을 받아옵니다.
# .content: 응답 객체에서 텍스트 내용(답변)만 추출하여 출력합니다.
print("\n--- [4] OpenAI 모델 응답 ---")
print(llm1.invoke(prompt_text).content)


# =============================================================================
# [5] 타사 모델(HuggingFace) 연동 (Third-party Model Integration)
# =============================================================================
# 두 번째 LLM 객체 생성 (HuggingFace API를 OpenAI 인터페이스처럼 사용)
# base_url: HuggingFace의 추론 API 주소로 엔드포인트를 변경
# api_key: HuggingFace API 토큰 사용
# model: HuggingFace Hub에 호스팅된 특정 모델 지정 (여기서는 SmolLM3-3B 모델 사용)
llm2 = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HUGGINGFACEHUB_API_TOKEN,
    model="HuggingFaceTB/SmolLM3-3B",
    temperature=0
)

# 동일한 질문 준비
prompt_text_2 = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"

# llm2를 통해 HuggingFace 모델의 응답을 받아 출력합니다.
print("\n--- [5] HuggingFace 모델 응답 ---")
print(llm2.invoke(prompt_text_2).content)


# =============================================================================
# [6] ModelLaboratory를 이용한 모델 비교 (Model Comparison)
# =============================================================================
# ModelLaboratory 객체 생성: 비교하고 싶은 LLM 객체들을 리스트로 전달합니다.
model_lab = ModelLaboratory.from_llms([llm1, llm2])

# compare 메서드: 동일한 질문을 등록된 모든 모델에게 보내고 결과를 비교하여 출력합니다.
print("\n--- [6] 모델 비교 결과 (ModelLaboratory) ---")
model_lab.compare("대한민국의 가을은 몇 월부터 몇 월까지야?")


# =============================================================================
# [7] Output Parser 활용 (Output Parsing)
# =============================================================================
# 출력을 쉼표로 구분된 리스트 형태로 파싱해주는 파서 객체 생성
output_parser = CommaSeparatedListOutputParser()

# 파서가 요구하는 지시사항(프롬프트에 포함될 가이드)을 가져옵니다.
# 예: "Your response should be a list of comma separated values, ..."
format_instructions = output_parser.get_format_instructions()

# LLM 객체 재설정 (max_tokens 제한 추가)
llm = ChatOpenAI(temperature=0, max_tokens=2048, model_name='gpt-4o-mini')

# 파서 지시사항을 포함한 새로운 프롬프트 템플릿 생성
# partial_variables: 템플릿을 완성하기 전에 미리 포맷 지시사항 등을 주입해둘 때 사용
prompt = PromptTemplate(
    template="10개의 팀을 보여줘 {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

# 사용자의 질문 주제 정의
query = "한국의 야구팀은?"

# 1. 템플릿에 주제(query)를 넣어 완성된 프롬프트를 만듭니다.
# 2. LLM에게 전달하여 결과를 받습니다 (.invoke)
# 3. 결과에서 텍스트 내용만 추출합니다 (.content)
output = llm.invoke(prompt.format(subject=query)).content

# 파서를 사용하여 문자열 응답을 실제 파이썬 리스트 자료형으로 변환합니다.
parsed_result = output_parser.parse(output)

print("\n--- [7] 파싱된 결과 (Python List) ---")
print(parsed_result) # 예: ['두산 베어스', 'LG 트윈스', ...]