# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os
import json
import langchain
from dotenv import load_dotenv

# LangChain Hub: 검증된 프롬프트(Prompt)를 다운로드하여 사용할 수 있는 저장소
from langchain_classic import hub

# Agent 관련 모듈 임포트
# AgentExecutor: 에이전트를 실행하는 런타임(Runtime) 환경
# create_react_agent: ReAct 방식(Reasoning + Acting)의 에이전트 생성 함수
# load_tools: LangChain에서 제공하는 기본 도구(Tools)들을 로드하는 함수
from langchain_classic.agents import AgentExecutor, create_react_agent, load_tools

# OpenAI의 채팅 모델(GPT-3.5, GPT-4 등)을 사용하기 위한 클래스
from langchain_openai import ChatOpenAI

# 사용자 정의 도구(Custom Tool)를 생성하기 위한 클래스
from langchain_classic.tools import Tool

# 프롬프트 템플릿: 변수를 사용하여 동적으로 프롬프트를 생성하는 도구
from langchain_classic.prompts import PromptTemplate

# LLMChain: 프롬프트와 LLM을 연결하여 실행하는 가장 기본적인 체인
from langchain_classic import LLMChain

# OpenAI 전용 에이전트 생성 함수들
# create_openai_functions_agent: OpenAI의 Function Calling 기능을 활용하는 에이전트
# create_openai_tools_agent: OpenAI의 최신 Tools 기능을 활용하는 에이전트 (권장)
from langchain_classic.agents import create_openai_functions_agent, create_openai_tools_agent 

# 정형화된 데이터 추출을 위한 체인 생성 함수
from langchain_classic.chains import create_extraction_chain

# LLM 답변의 품질을 평가하기 위한 로더
from langchain_classic.evaluation import load_evaluator


# .env 파일에서 환경 변수 로드 (API Key 보안 관리)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# LangChain의 내부 로그를 상세하게 출력하도록 설정 (디버깅용)
langchain.verbose = True


# =============================================================================
# [2] ReAct 에이전트 & 터미널 도구 (ReAct Agent with Terminal Tool)
# =============================================================================
# ReAct(Reasoning + Acting)는 LLM이 생각(Reasoning)하고 행동(Acting)하고 관찰(Observation)하는 과정을 반복하며 문제를 해결합니다.

# 1. LLM 초기화: 온도를 0으로 설정하여 결정적인(사실적인) 답변 유도
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 2. 도구 로드: 'terminal' 도구 사용
# allow_dangerous_tools=True: 터미널 명령어는 시스템에 영향을 줄 수 있으므로 명시적 허용 필요
tools = load_tools(["terminal"], allow_dangerous_tools=True)

# 3. 프롬프트 가져오기: LangChain Hub에서 ReAct 방식의 표준 프롬프트 다운로드
prompt = hub.pull("hwchase17/react")

# 4. 에이전트 생성: LLM, 도구, 프롬프트를 결합하여 ReAct 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

# 5. 에이전트 실행기(Executor) 생성: 에이전트가 실제로 도구를 사용하고 결과를 처리하도록 함
# verbose=True: 실행 과정을 상세히 출력
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 에이전트 실행
# 사용자의 요청: sample_data 디렉터리 확인 및 인코딩 처리 요청
result = agent_executor.invoke({"input": "sample_data 디렉터리에 있는 파일 목록을 알려줘, 터미널을 사용할 때 인코딩도 알아서 맞춰줘"})

print(result["output"])


# =============================================================================
# [3] 사용자 정의 도구 (Custom Tools) - The Answer
# =============================================================================
# 개발자가 직접 만든 함수를 에이전트가 사용할 수 있는 도구로 변환합니다.

# 에이전트가 호출할 실제 파이썬 함수 정의
def my_super_func(param):
    return "42"

# Tool 객체 리스트 생성
tools = [
    Tool.from_function(
        func=my_super_func,            # 실행할 함수
        name="The_Answer",             # 도구의 이름 (LLM이 선택할 때 사용)
        description="생명, 우주, 그리고 모든 것에 대한 궁극적인 질문의 답" # 도구 설명 (LLM의 선택 기준)
    ),
]

# 에이전트 및 실행기 생성
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 에이전트 실행: "이 세계의 진리"를 물어보면 description을 보고 "The_Answer" 도구를 선택함
result = agent_executor.invoke({"input": "이 세계의 진리를 알려주세요"})
print(result)


# =============================================================================
# [4] LangChain 체인을 도구로 활용 (Chain as a Tool) - Summarizer
# =============================================================================
# 단순히 함수뿐만 아니라, LangChain의 체인(Chain) 자체를 하나의 도구로 만들어 에이전트에 부여할 수 있습니다.

# 1. 요약 템플릿 정의
summarize_template = """아래의 글을 결론만 한 마디로 요약해 주세요.
{input}
"""

# 2. 프롬프트 템플릿 객체 생성
summarize_prompt = PromptTemplate(
    input_variables=["input"],
    template=summarize_template,
)

# 3. 요약 전용 체인 생성 (LLM + Prompt)
chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

# 4. 체인을 도구(Tool)로 래핑
# 에이전트가 "요약"이 필요할 때 이 도구를 사용하게 됨
tools = [
    Tool.from_function(
        func=summarize_chain.run,  # 체인의 실행 메서드를 연결
        name="Summarizer",         # 도구 이름
        description="Text summarizer" # 도구 설명
    ),
]


# 5. 에이전트 생성 및 실행
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 요약할 대상 텍스트
text = """다음을 요약해 주세요.
안녕하세요! 저는 ChatGPT라고 불리는 AI 언어 모델입니다.
OpenAI가 개발한 GPT-3.5 아키텍처를 기반으로 합니다.
저는 자연어 이해와 생성을 전문으로 하며, 다양한 주제에 대한 질문에 답하거나, 대화를 나누는 것을 잘합니다.
제 트레이닝 데이터는 2021년 9월까지의 정보를 기반으로 하기 때문에, 그 이후의 사건에 대해서는 지식이 없습니다.
하지만, 가능한 한 도움을 드리기 위해 노력할 것입니다.
질문이나 대화, 정보 공유 등, 어떤 도움이든 편하게 말씀해 주세요! 잘 부탁드립니다."""

# 에이전트 실행: 텍스트를 입력으로 주면 "Summarizer" 도구를 호출하여 요약 수행
result = agent_executor.invoke({"input": text})
print(result)


# =============================================================================
# [5] OpenAI Functions Agent (Terminal)
# =============================================================================
# OpenAI 모델의 'Function Calling' 기능을 활용한 에이전트입니다.
# ReAct 방식보다 더 구조적이고 안정적으로 도구를 호출할 수 있습니다.

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
tools = load_tools(["terminal"], allow_dangerous_tools=True)

# OpenAI Functions 전용 프롬프트 다운로드
prompt = hub.pull("hwchase17/openai-functions-agent")


# 에이전트 생성 (create_openai_functions_agent 사용)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행: 터미널 도구를 사용하여 파일 목록 확인
agent_executor.invoke({"input": "sample_data 디렉터리에 있는 파일 목록을 알려줘, 터미널을 사용할 때 인코딩도 알아서 맞춰서 해줘"})


# =============================================================================
# [6] OpenAI Functions Agent (Search)
# =============================================================================
# 검색 엔진(DuckDuckGo) 도구를 사용하는 에이전트 설정

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# 'ddg-search': DuckDuckGo 검색 도구 로드 (인터넷 검색 가능)
tools = load_tools(["ddg-search"])
prompt = hub.pull("hwchase17/openai-functions-agent")


# 에이전트 생성
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 실행: 서울과 부산의 날씨 검색
# 에이전트가 "서울 날씨", "부산 날씨"를 각각 검색하거나 한 번에 검색하여 정보를 종합함
result = agent_executor.invoke({"input": "서울과 부산의 날씨를 알려줘"})


print(result["output"])


# =============================================================================
# [7] OpenAI Tools Agent (Recommended)
# =============================================================================
# OpenAI의 최신 'Tools' API를 사용하는 에이전트입니다.
# 현재 가장 권장되는 방식으로, 병렬 함수 호출 등 복잡한 기능을 지원합니다.

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
tools = load_tools(["ddg-search"])

# Tools Agent 전용 프롬프트
prompt = hub.pull("hwchase17/openai-tools-agent")


# 에이전트 생성 (create_openai_tools_agent 사용)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행: 동일한 날씨 검색 작업을 최신 Tools 방식으로 수행
result = agent_executor.invoke({"input": "서울과 부산의 날씨를 알려줘"})

print(result["output"])


# =============================================================================
# [8] 구조화된 데이터 추출 (Structured Output Extraction)
# =============================================================================
# 비정형 텍스트(자연어)에서 원하는 정보만 추출하여 정형 데이터(JSON)로 변환합니다.

# 1. 추출할 데이터의 스키마(형식) 정의
schema = {
    "properties": {
        "person_name": {"type": "string"},      # 사람 이름
        "person_height": {"type": "integer"},   # 키 (정수)
        "person_hair_color": {"type": "string"},# 머리색
        "dog_name": {"type": "string"},         # 강아지 이름
        "dog_breed": {"type": "string"},        # 견종
    },
    "required": ["person_name", "person_height"], # 필수 필드
}


# 입력 텍스트: 사람과 강아지에 대한 정보가 섞여 있는 자연어 문장
text = """
Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek.
"""

chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 2. 추출 체인 생성: 스키마와 LLM을 연결
chain = create_extraction_chain(schema, chat)

# 3. 실행: 텍스트를 분석하여 스키마에 맞는 리스트 형태의 데이터 반환
people = chain.invoke(text)

# 결과 출력 (JSON 포맷으로 보기 좋게 출력)
print(json.dumps(people, indent=2))


# =============================================================================
# [9] 결과 평가 (Evaluation)
# =============================================================================
# LLM이 생성한 답변이 정답(Reference)과 일치하는지, 논리적으로 맞는지 평가합니다.

chat = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 1. 평가자(Evaluator) 로드: 'qa' (질의응답) 유형
evaluator = load_evaluator("qa", eval_llm=chat)

# 2. 평가 실행
result = evaluator.evaluate_strings(
    # 질문 (Input)
    input="""
    나는 시장에 가서 사과 10개를 샀어.
    사과 2개를 이웃에게 주고, 2개를 수리공에게 주었어.
    그리고 사과 5개를 더 사서 1개는 내가 먹었어.
    나는 몇 개의 사과를 가지고 있었니?
    """,
    # 모델의 답변 (Prediction) - 미리 생성된 답변이라고 가정
    prediction="""
    먼저 사과 10개로 시작했어.
    이웃에게 2개, 수리공에게 2개를 나누어 주었으므로 사과가 6개가 남았어.
    그런 다음 사과 5개를 더 사서 이제 사과가 11개가 되었어.
    마지막으로 사과 1개를 먹었으므로 사과 10개가 남게 돼.
    """,
    # 정답 (Reference)
    reference="10개",
)

# 평가 결과 출력 (CORRECT/INCORRECT, 점수 등)
print(result)