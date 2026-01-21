# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os
import langchain
import json
from dotenv import load_dotenv

# LangChain Hub: 검증된 프롬프트(Prompt)를 중앙 저장소에서 다운로드하여 사용
from langchain_classic import hub

# 에이전트(Agent) 관련 모듈 임포트
# - AgentExecutor: 에이전트를 실행하는 런타임 환경 (행동 결정 -> 도구 실행 -> 결과 관찰 반복)
# - create_react_agent: ReAct(Reasoning + Acting) 방식의 에이전트 생성
# - load_tools: LangChain에서 기본 제공하는 도구(Tools) 로드 (예: 검색, 계산기, 터미널 등)
# - create_openai_functions_agent: OpenAI의 Function Calling API를 사용하는 에이전트 생성
# - create_openai_tools_agent: OpenAI의 최신 Tools API를 사용하는 에이전트 생성 (권장 방식)
from langchain_classic.agents import AgentExecutor, create_react_agent, load_tools, create_openai_functions_agent, create_openai_tools_agent

# OpenAI의 채팅 모델(GPT-3.5, GPT-4 등)을 사용하기 위한 클래스
from langchain_openai import ChatOpenAI

# 사용자 정의 도구(Custom Tool)를 생성하기 위한 클래스
from langchain_classic.tools import Tool

# 프롬프트 템플릿: 변수({var})를 사용하여 동적으로 프롬프트 텍스트를 생성하는 도구
from langchain_classic.prompts import PromptTemplate

# LLMChain: 프롬프트와 LLM을 연결하여 순차적으로 실행하는 가장 기본적인 체인
# (참고: 최신 버전에서는 LCEL 스타일 권장하지만 레거시 호환성을 위해 유지)
from langchain_classic import LLMChain

# 정형화된 데이터 추출을 위한 체인 생성 함수
# (참고: create_extraction_chain은 deprecated 되었으며 with_structured_output 사용 권장됨)
from langchain_classic.chains import create_extraction_chain

from langchain_classic.evaluation import load_evaluator


# .env 파일 로드 (환경 변수 설정)
# 일반적으로 API Key(.env에 저장됨)를 로드하기 위해 사용
load_dotenv()


# 환경 변수에서 OpenAI API Key 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# LangChain의 내부 동작 로그를 상세하게 출력하도록 설정 (디버깅 시 유용)
# verbose=True로 설정하면 체인이나 에이전트의 사고 과정(Thought Process)이 출력됨
langchain.verbose = True



# =============================================================================
# [2] ReAct 에이전트 & 터미널 도구 (ReAct Agent with Terminal Tool)
# =============================================================================
# ReAct 방식은 LLM이 [생각 -> 행동 -> 관찰] 루프를 돌며 문제를 해결함

# 1. LLM 초기화 (GPT-4.1-mini 사용, temperature=0으로 설정하여 무작위성 제거/일관된 답변 유도)
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 2. 도구(Tool) 로드
# "terminal": 시스템 터미널 명령어를 실행할 수 있는 도구
# allow_dangerous_tools=True: 터미널 명령어 실행은 시스템에 영향을 줄 수 있으므로 위험한 도구 사용 허용 플래그 필수
tools = load_tools(["terminal"], allow_dangerous_tools=True)

# 3. 프롬프트 로드
# LangChain Hub에서 ReAct 에이전트용 표준 프롬프트("hwchase17/react")를 다운로드
prompt = hub.pull("hwchase17/react")

# 4. 에이전트 생성 (LLM + Tools + Prompt 결합)
agent = create_react_agent(llm, tools, prompt)

# 5. 에이전트 실행기(Executor) 생성
# verbose=True: 실행 중간 과정을 터미널에 출력
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 에이전트 실행
# 사용자의 요청: sample_data 디렉터리 확인 및 인코딩 처리
# 해당 폴더가 없을 경우에 대한 예외 처리 요청도 포함됨
result = agent_executor.invoke({"input": "sample_data 디렉터리에 있는 파일 목록을 알려줘, 터미널을 사용할 때 인코딩도 알아서 맞춰줘, 해당 폴더가 없으면 없다고 해줘"})

# 실행 결과 출력 (최종 답변)
print(result["output"])



# =============================================================================
# [3] 사용자 정의 도구 (Custom Tools) - The Answer
# =============================================================================
# 파이썬 함수를 정의하고, 이를 에이전트가 사용할 수 있는 도구(Tool)로 변환하는 예제

# 1. 실제 로직을 수행할 파이썬 함수 정의
def my_super_func(param):
    return "42"

# 2. 도구 목록(tools)에 사용자 정의 함수 추가
# Tool.from_function을 사용하여 함수를 래핑(Wrapping)
tools = [
    Tool.from_function(
        func=my_super_func,            # 실행할 함수
        name="The_Answer",             # 도구 이름 (LLM이 이 이름을 보고 도구를 선택)
        description="생명, 우주, 그리고 모든 것에 대한 궁극적인 질문의 답" # 도구 설명 (매우 중요: LLM이 언제 이 도구를 써야 할지 판단하는 기준)
    ),
]


# 3. 에이전트 재생성 (새로운 도구 리스트 적용)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. 실행: 질문("이 세계의 진리")이 description과 매칭되어 "The_Answer" 도구가 호출됨
result = agent_executor.invoke({"input": "이 세계의 진리를 알려주세요"})
print(result)





# =============================================================================
# [4] LangChain 체인을 도구로 활용 (Chain as a Tool) - Summarizer
# =============================================================================
# 단순 함수뿐만 아니라, 또 다른 LLM 체인(Chain)을 하나의 도구로 만들어 에이전트에 장착 가능

# 1. 요약(Summarize)을 위한 프롬프트 템플릿 정의
summarize_template = """
아래의 글을 결론만 한 마디로 요약해 주세요.
{input}
"""

summarize_prompt = PromptTemplate(
    input_variables=["input"],
    template=summarize_template,
)

# 2. 요약 작업을 수행할 LLM 체인 생성
chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

# 3. 체인을 도구로 변환하여 리스트에 저장
tools = [
    Tool.from_function(
        func=summarize_chain.run,  # 체인의 실행 메서드(run)를 도구의 기능으로 설정
        name="Summarizer",         # 도구 이름
        description="Text summarizer" # 도구 설명: 텍스트 요약이 필요할 때 사용됨
    ),
]


# 4. 에이전트 생성 및 실행
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 요약할 대상 텍스트 (ChatGPT 소개글)
text = """
다음을 요약해 주세요.
안녕하세요! 저는 ChatGPT라고 불리는 AI 언어 모델입니다.
OpenAI가 개발한 GPT-3.5 아키텍처를 기반으로 합니다.
저는 자연어 이해와 생성을 전문으로 하며, 다양한 주제에 대한 질문에 답하거나, 대화를 나누는 것을 잘합니다.
제 트레이닝 데이터는 2021년 9월까지의 정보를 기반으로 하기 때문에, 그 이후의 사건에 대해서는 지식이 없습니다.
하지만, 가능한 한 도움을 드리기 위해 노력할 것입니다.
질문이나 대화, 정보 공유 등, 어떤 도움이든 편하게 말씀해 주세요! 잘 부탁드립니다.
"""

# 실행: 입력 텍스트를 보고 에이전트가 'Summarizer' 도구를 호출하여 요약 결과 반환
result = agent_executor.invoke({"input": text})
print(result)




# =============================================================================
# [5] OpenAI Functions Agent (Terminal Tool)
# =============================================================================
# OpenAI 모델 특화 기능인 'Function Calling'을 사용하는 에이전트
# ReAct 방식보다 JSON 구조를 통해 더 정확하고 안정적으로 도구를 호출함

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
tools = load_tools(["terminal"], allow_dangerous_tools=True)

# OpenAI Functions Agent 전용 프롬프트 다운로드 (hwchase17/openai-functions-agent)
prompt = hub.pull("hwchase17/openai-functions-agent")

# 에이전트 생성 (create_openai_functions_agent 사용)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행: 터미널 도구를 사용하여 파일 목록 확인 요청
agent_executor.invoke({"input": "sample_data 디렉터리에 있는 파일 목록을 알려줘, 터미널을 사용할 때 인코딩도 알아서 맞춰서 해줘"})




# =============================================================================
# [6] 디버깅 및 로깅 설정 (Debugging & Logging)
# # =============================================================================
# LangChain의 디버그 모드를 활성화하여 모든 입출력을 상세히 추적
langchain.debug = True
# verbose를 False로 설정하여 중복 출력 방지 (debug 모드가 더 상세함)
langchain.verbose = False




# =============================================================================
# [7] OpenAI Functions Agent (Search Tool)
# =============================================================================
# DuckDuckGo 검색 도구를 사용하여 외부 정보를 검색하는 에이전트

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# 'ddg-search': DuckDuckGo 검색 엔진 툴 로드
tools = load_tools(["ddg-search"])
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행: 서울과 부산의 날씨를 검색 (검색 툴이 여러 번 호출될 수 있음)
result = agent_executor.invoke({"input": "서울과 부산의 날씨를 알려줘"})

print(result["output"])





# =============================================================================
# [8] OpenAI Tools Agent (Recommended)
# =============================================================================
# OpenAI의 최신 API인 'Tools'를 사용하는 에이전트 (Function Calling의 개선된 버전)
# 병렬 함수 호출(Parallel Function Calling) 등을 지원하여 더 효율적임

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
tools = load_tools(["ddg-search"]) # 검색 툴 로드
# Tools Agent 전용 프롬프트 다운로드
prompt = hub.pull("hwchase17/openai-tools-agent")


# 에이전트 생성 (create_openai_tools_agent 사용)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 실행: 동일한 날씨 검색 요청을 최신 Tools 에이전트로 처리
result = agent_executor.invoke({"input": "서울과 부산의 날씨를 알려줘"})


print(result["output"])





# =============================================================================
# [9] 구조화된 데이터 추출 (Extraction Chain with Legacy Method)
# =============================================================================
# 비정형 텍스트에서 특정 스키마에 맞는 데이터를 추출하는 예제
# 현재 코드에서는 deprecated된 'create_extraction_chain'을 사용하고 있음
# (최신 방식은 ChatModel의 '.with_structured_output' 메서드 사용 권장)

# 1. 추출할 데이터 형식을 JSON Schema로 정의
schema = {
    "properties": {
        "person_name": {"type": "string"},      # 사람 이름 (문자열)
        "person_height": {"type": "integer"},   # 키 (정수)
        "person_hair_color": {"type": "string"},# 머리색 (문자열)
        "dog_name": {"type": "string"},         # 강아지 이름 (문자열)
        "dog_breed": {"type": "string"},        # 견종 (문자열)
    },
    "required": ["person_name", "person_height"], # 필수 포함 필드 지정
}

# 2. 정보를 추출할 대상 텍스트 (자연어 문장)
text = """
Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek.
"""

chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 3. 추출 체인 생성 (스키마 + LLM)
chain = create_extraction_chain(schema, chat)

# 4. 실행 및 결과 저장
people = chain.invoke(text)

# 결과 출력
print(json.dumps(people, indent=2))
# (참고: 실행 시 deprecated 경고가 발생할 수 있음)



# =============================================================================
# [10] 결과 평가 (Evaluation) - QA Evaluator
# =============================================================================
# LLM이 생성한 답변의 품질을 자동으로 평가하는 기능입니다.
# 정답(Reference)이 있는 경우, 모델의 답변(Prediction)이 얼마나 정확한지 판단할 때 사용합니다.

chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 1. 평가자(Evaluator) 로드
# "qa": 질의응답(Question Answering) 유형의 평가를 수행하는 evaluator를 로드합니다.
# eval_llm: 평가를 수행할 LLM 모델. (GPT-4와 같은 고성능 모델을 사용하는 것이 일반적이지만, 여기서는 gpt-4.1-mini 사용)
evaluator = load_evaluator("qa", eval_llm=chat)

# 2. 평가 실행 (evaluate_strings)
# input: 사용자의 원래 질문 또는 문제 상황
# prediction: 평가 대상 모델이 내놓은 답변 (AI의 현재 출력)
# reference: 정답 또는 모범 답안 (Ground Truth)
result = evaluator.evaluate_strings(
    input="""
        나는 시장에 가서 사과 10개를 샀어.
        사과 2개를 이웃에게 주고, 2개를 수리공에게 주었어.
        그리고 사과 5개를 더 사서 1개는 내가 먹었어.
        나는 몇 개의 사과를 가지고 있었니?
    """,
    prediction="""
        먼저 사과 10개로 시작했어.
        이웃에게 2개, 수리공에게 2개를 나누어 주었으므로 사과가 6개가 남았어.
        그런 다음 사과 5개를 더 사서 이제 사과가 11개가 되었어.
        마지막으로 사과 1개를 먹었으므로 사과 10개가 남게 돼.
    """,
    reference="10개",
)

# 3. 평가 결과 출력
# 결과는 딕셔너리 형태로 반환되며, 보통 다음과 같은 키를 포함합니다.
# - reasoning: 평가 모델이 왜 맞거나 틀렸다고 판단했는지에 대한 추론 과정
# - value: 'CORRECT' 또는 'INCORRECT' (정답 여부)
# - score: 점수 (1 또는 0)
print(result)