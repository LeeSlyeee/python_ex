# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# LangChain 라이브러리 임포트
# -----------------------------------------------------------------------------
# LangChain은 LLM(대규모 언어 모델)을 활용한 애플리케이션 개발을 돕는 프레임워크입니다.
# langchain_openai: OpenAI 모델(GPT-3.5, GPT-4 등)과 상호작용하기 위한 패키지
from langchain_openai import ChatOpenAI

# langchain_classic.schema: 대화형 모델에서 사용되는 메시지 객체들
# - AIMessage: AI(Assistant)가 생성한 응답 메시지
# - HumanMessage: 사용자(User)가 입력한 질문이나 요청 메시지
# - SystemMessage: 대화의 맥락이나 AI의 페르소나(역할)를 설정하는 시스템 메시지
from langchain_classic.schema import AIMessage, HumanMessage, SystemMessage

# langchain_classic.callbacks: LLM 실행 중 발생하는 이벤트(예: 토큰 생성)를 처리하는 콜백
# - StreamingStdOutCallbackHandler: 생성되는 텍스트를 실시간으로 콘솔(표준 출력)에 스트리밍하는 핸들러
from langchain_classic.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# langchain_classic.prompts: 동적인 프롬프트 생성을 위한 템플릿 도구들
# - PromptTemplate: 일반적인 텍스트 기반의 프롬프트 템플릿
# - ChatPromptTemplate: 대화형 모델(Chat Model)을 위한 메시지 리스트 기반 템플릿
# - SystemMessagePromptTemplate, HumanMessagePromptTemplate: 각 역할별 메시지 템플릿
from langchain_classic.prompts import PromptTemplate
from langchain_classic.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 데이터 유효성 검사 및 구조 정의 (Pydantic)
# - BaseModel: 데이터 구조(스키마)를 정의하는 기본 클래스
# - Field: 필드에 대한 설명(description)이나 제약 조건을 설정
from pydantic import BaseModel, Field

# langchain_classic.output_parsers: LLM의 텍스트 출력을 구조화된 데이터(예: JSON, 객체)로 변환
# - PydanticOutputParser: Pydantic 모델을 기반으로 출력을 파싱하는 파서
from langchain_classic.output_parsers import PydanticOutputParser

# langchain_classic.chains: 여러 컴포넌트(Prompt, LLM, Parser 등)를 연결하여 실행하는 체인
# - LLMChain: 가장 기본적인 체인 (Prompt + LLM)
# - SimpleSequentialChain: 하나의 체인 출력을 다음 체인의 입력으로 연결하는 순차적 체인
# - ConversationChain: 대화의 기억(Memory)을 관리하며 대화를 이어가는 체인
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SimpleSequentialChain
from langchain_classic.chains import ConversationChain

# langchain_classic.memory: 대화 내용을 저장하고 관리하는 메모리 모듈
# - ConversationBufferMemory: 대화의 모든 기록을 그대로 저장하여 프롬프트에 포함시키는 메모리
from langchain_classic.memory import ConversationBufferMemory

# .env 파일 활성화 (환경변수 로드)
# 프로젝트 루트 경로에 있는 .env 파일에서 OPENAI_API_KEY 등을 로드합니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# [2] 기본 텍스트 생성 (Basic Generation)
# =============================================================================
# ChatOpenAI 클래스를 인스턴스화하여 GPT 모델 객체를 생성합니다.
# - model_name="gpt-4.1-mini": 사용할 OpenAI 모델 지정 (실제 존재하는 모델명을 사용해야 함, 예: gpt-3.5-turbo, gpt-4)
# - temperature=0: 모델의 창의성(Randomness) 조절. 0이면 가장 결정적이고 사실적인 답변, 1에 가까울수록 창의적이고 다양한 답변 생성.
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# invoke 메서드: 모델에 입력을 전달하고 결과를 받아옵니다.
# 가장 단순한 문자열 입력 -> 문자열(또는 메시지) 출력 방식입니다.
result = llm.invoke("자기소개를 해주세요.")
print("--- [2] 기본 생성 결과 ---")
print(result)


# =============================================================================
# [3] 역할 기반 대화 (Role-based Chat)
# =============================================================================
# 대화형 모델(Chat Model)은 단일 문자열 대신 메시지 객체의 리스트를 입력으로 받습니다.
# 이를 통해 시스템 설정(System), 사용자 질문(Human), AI 답변(AI)의 역할을 명확히 구분합니다.

chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 메시지 리스트 구성
messages = [
    # SystemMessage: AI에게 "친절한 조수"라는 역할(Persona)을 부여
    SystemMessage(content="You are a helpful assistant."),
    
    # HumanMessage: 사용자가 "존"이라고 자기소개를 함
    HumanMessage(content="안녕하세요! 저는 존이라고 합니다!"),
    
    # AIMessage: AI가 이전에 이렇게 대답했다고 가정 (Few-shot Learning 또는 문맥 주입)
    # AI가 이미 사용자의 이름을 인지하고 있음을 맥락에 포함시킴
    AIMessage(content="안녕하세요, 존 씨! 어떻게 도와드릴까요?"),
    
    # HumanMessage: 사용자의 후속 질문. 앞선 맥락(자신의 이름)을 기억하는지 테스트
    HumanMessage(content= "제 이름을 아세요?")
]

# invoke 메서드에 메시지 리스트 전달
response = chat.invoke(messages)
print("\n--- [3] 역할 기반 대화 결과 ---")
print(response)


# =============================================================================
# [4] 스트리밍 출력 (Streaming Output)
# =============================================================================
# 긴 답변을 기다리지 않고, 생성되는 토큰 단위로 즉시 출력(Streaming)하는 방식입니다.
# - streaming=True: 스트리밍 모드 활성화
# - callbacks=[StreamingStdOutCallbackHandler()]: 생성된 토큰을 표준 출력(콘솔)에 바로 찍어주는 핸들러 등록

chat = ChatOpenAI(
    model_name="gpt-4.1-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

messages = [HumanMessage(content="자기소개를 해주세요.")]

print("\n--- [4] 스트리밍 출력 시작 ---")
result = chat.invoke(messages)
print("\n--- [4] 스트리밍 출력 종료 ---")


# =============================================================================
# [5] 프롬프트 템플릿 - 문자열 기반 (Prompt Templates - String)
# =============================================================================
# 반복적으로 사용되는 프롬프트 형식을 템플릿화하여 변수만 바꿔가며 재사용합니다.

# 템플릿 문자열 정의 ({dish} 부분이 변수로 치환됨)
template = """
다음 요리의 레시피를 생각해 주세요.
요리: {dish}
"""

# PromptTemplate 객체 생성
# - input_variables: 템플릿 내에서 사용할 변수 리스트
# - template: 실제 프롬프트 템플릿 문자열
prompt = PromptTemplate(
    input_variables=["dish"],
    template=template,
)

# format 메서드: 변수에 실제 값("카레")을 넣어 완성된 프롬프트 문자열 생성
result = prompt.format(dish="카레")
print("\n--- [5] 문자열 프롬프트 템플릿 결과 ---")
print(result)


# =============================================================================
# [6] 프롬프트 템플릿 - 메시지 기반 (Chat Prompt Templates)
# =============================================================================
# 채팅 모델에 특화된 템플릿으로, System/Human 메시지 구조를 유지하면서 내용을 동적으로 변경합니다.

# ChatPromptTemplate 생성
# - from_messages: 메시지 템플릿들의 리스트로부터 생성
chat_prompt = ChatPromptTemplate.from_messages([
    # 시스템 메시지 템플릿: {country} 변수 사용
    SystemMessagePromptTemplate.from_template("당신은 {country} 요리 전문가입니다."),
    # 사용자 메시지 템플릿: {dish} 변수 사용
    HumanMessagePromptTemplate.from_template("다음 요리의 레시피를 생각해 주세요.\n\n요리: {dish}")
])

# format_prompt: 변수(country="영국", dish="고기감자조림")를 할당하여 PromptValue 객체 생성
# to_messages: PromptValue를 실제 메시지 객체 리스트([SystemMessage, HumanMessage])로 변환
messages = chat_prompt.format_prompt(country="영국", dish="고기감자조림").to_messages()

print("\n--- [6] 채팅 프롬프트 메시지 구성 ---")
print(messages)

# 완성된 메시지로 LLM 호출
chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
result = chat.invoke(messages)
print("\n--- [6] LLM 응답 결과 ---")
print(result.content)


# =============================================================================
# [7] 정형화된 출력 파싱 - 준비 (Output Parsers: Setup)
# =============================================================================
# LLM의 출력을 단순 텍스트가 아닌, 프로그램에서 사용하기 쉬운 JSON 등의 구조화된 데이터로 받기 위한 설정입니다.

# 1. Pydantic을 이용한 데이터 구조(Schema) 정의
class Recipe(BaseModel):
    # 재료 목록: 문자열 리스트
    ingredients: list[str] = Field(description="ingredients of the dish")
    # 조리 순서: 문자열 리스트
    steps: list[str] = Field(description="steps to make the dish")

# 2. Parser 생성: 정의한 Pydantic 모델을 기반으로 파서 인스턴스화
parser = PydanticOutputParser(pydantic_object=Recipe)

# 3. 포맷 지시사항(Format Instructions) 가져오기
# 파서가 "JSON 형식으로 출력하고, 키는 ingredients와 steps여야 한다"는 식의 프롬프트 안내문을 자동 생성합니다.
format_instructions = parser.get_format_instructions()
print("\n--- [7] 포맷 지시사항 (Format Instructions) ---")
print(format_instructions)


# =============================================================================
# [8] 정형화된 출력 파싱 - 적용 (Output Parsers: Application)
# =============================================================================
# 위에서 생성한 포맷 지시사항을 실제 프롬프트에 포함시켜 LLM에게 전달합니다.

# 프롬프트 템플릿에 {format_instructions} 변수 자리를 마련
template = """다음 요리의 레시피를 한국어로 생각해 주세요.

{format_instructions}

요리: {dish}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["dish"],
    # partial_variables: 'format_instructions'는 미리 채워두는 변수임 (실행 시마다 바뀌지 않으므로)
    partial_variables={"format_instructions": format_instructions}
)

# 프롬프트 완성
formatted_prompt = prompt.format(dish="카레")
print("\n--- [8] 완성된 프롬프트 ---")
print(formatted_prompt)

# LLM 호출
chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
messages = [HumanMessage(content=formatted_prompt)]
output = chat.invoke(messages)

print("\n--- [8] LLM 원본 응답 (Raw Text) ---")
print(output.content)

# 파싱: 텍스트 응답을 Recipe 객체로 변환
recipe = parser.parse(output.content)

print("\n--- [8] 파싱된 결과 객체 (Parsed Object) ---")
print(f"Type: {type(recipe)}") # <class '__main__.Recipe'>
print(recipe)


# =============================================================================
# [9] LLMChain 활용 (Using LLMChain)
# =============================================================================
# Prompt + LLM + OutputParser 과정을 하나의 '체인(Chain)'으로 묶어서 관리합니다.
# 이렇게 하면 매번 프롬프트를 포맷팅하고, 호출하고, 파싱하는 코드를 작성할 필요가 없습니다.

# Pydantic 모델 및 파서 재정의 (실습의 독립성을 위해)
class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

output_parser = PydanticOutputParser(pydantic_object=Recipe)

template = """다음 요리의 레시피를 한국어로 생각해 주세요.

{format_instructions}

요리: {dish}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["dish"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# LLMChain 생성
# - prompt: 입력 템플릿
# - llm: 사용할 언어 모델
# - output_parser: 출력을 처리할 파서
chain = LLMChain(prompt=prompt, llm=chat, output_parser=output_parser)

# 체인 실행 (invoke)
# 입력값으로 "카레"를 주면 내부적으로 [Prompt 포맷팅 -> LLM 호출 -> 파싱] 과정이 수행됨
recipe = chain.invoke("카레")

print("\n--- [9] LLMChain 실행 결과 ---")
print(f"Type: {type(recipe)}")
print(recipe)


# =============================================================================
# [10] 연속 체인 (SimpleSequentialChain)
# =============================================================================
# 두 개 이상의 체인을 순차적으로 연결합니다.
# Chain 1의 출력이 Chain 2의 입력으로 들어가는 파이프라인 구조입니다.

chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 1. 첫 번째 체인: 단계별 추론 (Chain of Thought)
# 질문을 받아 단계별로 생각하여 답을 도출하도록 유도
cot_template = """다음 질문에 답하세요.
질문: {question}
단계별로 생각해 봅시다.
"""
cot_prompt = PromptTemplate(
    input_variables=["question"],
    template=cot_template,
)
cot_chain = LLMChain(llm=chat, prompt=cot_prompt)


# 2. 두 번째 체인: 요약 (Summarization)
# 앞선 단계의 상세한 답변을 결론만 간단히 요약하도록 유도
summarize_template = """다음 문장을 결론만 간단히 요약하세요.
{input}
"""
summarize_prompt = PromptTemplate(
    input_variables=["input"],
    template=summarize_template,
)
summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

# 3. 체인 연결 (SimpleSequentialChain)
# chains 리스트에 순서대로 등록
cot_summarize_chain = SimpleSequentialChain(chains=[cot_chain, summarize_chain])

# 실행
# 복잡한 논리 퀴즈를 입력으로 전달
print("\n--- [10] 연속 체인 (CoT -> 요약) 실행 결과 ---")
result = cot_summarize_chain.invoke(
"""저는 시장에 가서 사과 10개를 샀습니다.
이웃에게 2개, 수리공에게 2개를 주었습니다.
그런 다음에 사과 5개를 더 사서 1개를 먹었습니다.
남은 개수는 몇 개인가요?"""
)

print(result["output"])


# =============================================================================
# [11] 메모리 기능 (Memory)
# =============================================================================
# ConversationChain은 기본적으로 '대화 기능'을 위한 체인입니다.
# Memory 컴포넌트를 장착하여 이전 대화 내용을 기억하게 합니다.

chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# ConversationChain 생성
# - memory=ConversationBufferMemory(): 대화 내용을 메모리 버퍼에 저장
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

print("\n--- [11] 대화형 봇 시작 (종료하려면 '끝' 입력) ---")
while True:
    user_message = input("You: ")
    if user_message == "끝":
        print("(대화 종료)")
        break
    
    # input을 전달하면 memory에 저장된 이전 대화 내용과 함께 LLM에 전달됨
    ai_message = conversation.invoke(input=user_message)["response"]
    print(f"AI: {ai_message}")


# =============================================================================
# [12] 최종 확인 (Final Check)
# =============================================================================
# 마지막으로 역할 기반 대화가 정상적으로 동작하는지 확인하는 테스트 코드

chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
messages = [
    SystemMessage(content="You are a helpful assistant."), # 시스템 설정
    HumanMessage(content="안녕하세요. 저는 존이라고 합니다!"),  # 사용자 입력
    AIMessage(content="안녕하세요, 존 님! 어떻게 도와드릴까요?"), # AI 맥락 주입
    HumanMessage(content= "제 이름을 아세요?")                # 사용자 확인 질문
]

result = chat.invoke(messages)
print("\n--- [12] 최종 확인 결과 ---")
print(result.content)