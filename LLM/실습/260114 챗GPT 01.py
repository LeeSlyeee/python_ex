# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os            # 운영체제 상호작용 (파일 경로, 환경변수 접근 등)
import json          # 데이터 직렬화/역직렬화 (OpenAI API의 JSON 응답 처리 및 함수 파라미터 파싱)
from dotenv import load_dotenv # .env 파일에서 환경변수를 로드하기 위한 라이브러리
from openai import OpenAI      # OpenAI API 서버와 통신하기 위한 클라이언트 객체

# .env 파일에 저장된 환경 변수(예: OPENAI_API_KEY)를 읽어와 시스템 환경변수로 등록
load_dotenv()

# 시스템 환경변수에서 'OPENAI_API_KEY' 값을 가져옴
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# [2] 기본 텍스트 생성 - 단일 메시지 (Basic Text Generation)
# =============================================================================
# OpenAI 클라이언트 인스턴스 초기화
client = OpenAI()

# 클라이언트를 통해 채팅 완료(Chat Completion) 요청 생성
response = client.chat.completions.create(
    model="gpt-4o-mini", # 사용할 모델 지정 (현재 가장 효율적인 소형 모델)
    messages=[
        {"role": "system", "content": "너는 GPT에 대한 전문가야."}, # 시스템 역할 부여 (페르소나 설정)
        {"role": "user", "content": "OpenAI 모델 목록에 'gpt-4.1-mini'가 있어?"} # 실제 사용자의 질문 또는 문구 전달
    ]
)

# 전체 응답 객체 출력
print("\n--- [1] 기본 응답 객체 ---")
print(response)

# 응답 객체를 JSON 형태의 문자열로 변환하여 보기 좋게 출력
print("\n--- [2] 기본 응답 (JSON 포맷) ---")
print(response.model_dump_json(indent=2))


# =============================================================================
# [3] 대화 맥락 유지 (Maintaining Chat Context)
# =============================================================================
# 이전 대화 내용을 포함하여 다시 요청 (모델이 이전 대화를 기억하게 함)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! I'm John."},
        {"role": "assistant", "content": "Hello John! How can I assist you today?"}, # 이전 AI의 응답
        {"role": "user", "content": "Do you know my name?"} # 사용자의 연속된 질문
    ]
)

print("\n--- [3] 문맥 파악 응답 ---")
print(response.model_dump_json(indent=2))


# =============================================================================
# [4] 스트리밍 응답 출력 (Streaming Response / Real-time Output)
# =============================================================================
# stream=True 옵션을 주면 응답이 생성되는 대로 한 글자/단어 단위(chunk)로 받아올 수 있음
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! I'm John."}
    ],
    stream=True # 스트리밍 모드 활성화
)

print("\n--- [4] 스트리밍 응답 시작 ---")
# 응답 객체가 이터레이터(Iterator) 형태이므로 for문으로 순회하며 내용 출력
for chunk in response:
    choice = chunk.choices[0]
    # finish_reason이 None이면 내용이 생성 중인 상태, content가 비어있지 않은 경우에만 출력
    if choice.finish_reason is None and choice.delta.content:
        print(choice.delta.content, end="") # 개행 없이 이어서 출력하여 실시간 효과 구현
print("\n--- 스트리밍 응답 종료 ---")


# =============================================================================
# [5] 함수 호출 - Function Calling (Tool Calling)
# =============================================================================
# 1. 외부 도구(함수) 정의
# 모델은 이 정보를 바탕으로 특정 상황에서 이 함수를 호출할지 결정함
def get_current_weather(location, unit="celsius"):
    """특정 위치의 실시간 날씨 정보를 반환하는 모의(Mock) 함수"""
    weather_info = {
        "location": location,
        "temperature": "25",
        "unit": "celsius",
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info) # 정보를 JSON 문자열로 변환하여 반환 (모델이 이해하기 쉬움)

# 2. 모델에게 전달할 함수 명세(Spec) 작성
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. Seoul",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"], # 필수 파라미터 지정
        },
    },
]

# 3. 대화 이력 리스트 생성 및 날씨 관련 질문 추가
messages = [
    {"role": "user", "content": "Hello! I'm John."},
    {"role": "assistant", "content": "Nice to meet you, John!"},
    {"role": "user", "content": "Do you know my name?"},
    {"role": "user", "content": "What is the weather in Seoul?"} # 함수 호출을 유도하는 질문
]

# 4. 모델에게 대화 내용과 사용 가능한 함수 목록 전달
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    functions=functions # 모델에게 사용할 수 있는 '도구'들을 알려줌
)

print("\n--- [5-1] 모델의 함수 호출 요청 (Function Call Result) ---")
print(response.model_dump_json(indent=2))

# 5. 모델의 응답 분석 및 실제 함수 실행
response_message = response.choices[0].message
available_functions = {
    "get_current_weather": get_current_weather,
}

# 모델이 함수 호출(function_call)을 요청했는지 확인
if response_message.function_call:
    function_name = response_message.function_call.name # 호출할 함수 이름 추출
    function_to_call = available_functions[function_name] # 실제 매핑된 파이썬 함수 객체 가져오기
    function_args = json.loads(response_message.function_call.arguments) # 모델이 생성한 인자 파싱

    # 파이썬 환경에서 실제 함수 실행
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

    print("\n--- [5-2] 실제 함수 실행 결과 ---")
    print(function_response)

    # 6. 함수 실행 결과를 대화 이력에 추가하여 다시 모델에게 전달
    # 모델은 이 결과를 바탕으로 사용자에게 줄 최종 문장을 생성함
    messages.append(response_message) # 모델의 함수 호출 요청 메시지 추가
    messages.append(
        {
            "role": "function", # 시스템이 함수 실행 결과를 주는 역할
            "name": function_name,
            "content": function_response,
        }
    )

    # 7. 함수의 결과값이 포함된 전체 대화 이력을 바탕으로 AI의 최종 답변 생성 요청
    second_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    print("\n--- [5-3] 함수의 결과를 반영한 AI의 최종 답변 ---")
    print(second_response.model_dump_json(indent=2))
else:
    print("\n--- [5-1] 모델이 함수 호출을 지시하지 않았습니다. ---")