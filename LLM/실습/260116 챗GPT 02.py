# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os            # 운영체제 인터페이스 (환경변수 로드 등)
import json          # JSON 데이터 처리
from dotenv import load_dotenv # .env 파일에서 환경변수 로드
from openai import OpenAI      # OpenAI API 클라이언트

# .env 파일 활성화 (API KEY 로드)
load_dotenv()

# OpenAI 클라이언트 인스턴스 생성 (환경변수의 OPENAI_API_KEY 자동 사용)
client = OpenAI()

# =============================================================================
# [2] 기본 텍스트 생성 및 시스템 프롬프트 활용 (Basic Generation & System Prompts)
# =============================================================================

# 1. 기본적인 질문 (User Role만 사용)
# - 별도의 지시사항 없이 단순한 정보 검색 질문
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "2020년 월드시리즈에서는 누가 우승했어?"}
    ]
)
print("--- [2-1] 기본 질문 결과 ---")
print(response.choices[0].message.content)


# 2. 페르소나(System Role) 부여
# - AI에게 '친절한 비서'라는 역할을 부여하고 '반말'로 대답하도록 지시
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "너는 친절하게 답변해주는 비서야. 답변은 반말로 해줘"},
        {"role": "user", "content": "2020년 월드시리즈에서는 누가 우승했어?"}
    ]
)
print("\n--- [2-2] 페르소나(반말 비서) 결과 ---")
print(response.choices[0].message.content)


# 3. 영어 답변 강제 (System Role 활용)
# - 시스템 프롬프트를 통해 반드시 영어로만 대답하도록 강력한 제약 설정
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You must only answer users' questions in English. This must be honored. You must only answer in English."},
        {"role": "user", "content": "2020년 월드시리즈에서는 누가 우승했어?"}
    ]
)
print("\n--- [2-3] 영어 답변 강제 결과 ---")
print(response.choices[0].message.content)


# 4. 번역 작업 수행
# - 질문에 답하지 않고, 입력된 텍스트를 번역하는 도구로서의 역할 수행
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "사용자의 질문에 답변을 하지말고 한글 입력을 영어로 번역하십시오."},
        {"role": "user", "content": "2020년 월드시리즈에서는 누가 우승했어?"}
    ]
)
print("\n--- [2-4] 단순 번역 결과 ---")
print(response.choices[0].message.content)


# =============================================================================
# [3] 대화 맥락 유지와 연속 대화 (Contextual Conversation)
# =============================================================================
# - 이전 대화(user-assistant)를 포함하여 후속 질문(Follow-up Question)을 던짐
# - "그 나라"가 어디인지 명시하지 않았지만, 앞선 대화 맥락을 통해 "한국"임을 유추
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "2002년 월드컵에서 가장 화제가 되었던 나라는 어디야?"},
        {"role": "assistant", "content": "바로 예상을 뚫고 4강 진출 신화를 일으킨 한국입니다."},
        {"role": "user", "content": "그 나라가 화제가 되었던 이유를 자세하게 설명해줘"}
    ]
)
print("\n--- [3] 연속 대화(맥락 파악) 결과 ---")
print(response.choices[0].message.content)


# =============================================================================
# [4] 정형화된 응답 유도 - 문서 요약 및 분석 (Structured Output: Analysis)
# =============================================================================
# - 시스템 프롬프트를 통해 출력 형식을 엄격하게 제어 (주제, 요약, 가능한 질문)
# - "가능한 질문"은 반드시 파이썬 리스트 포맷을 따르도록 지시

def return_answer_analysis(input_text=''):
    system_prompt = """특정 문서가 입력되면 다음과 같은 형태로 문서를 분석하십시오.
    1. 주어진 입력:에 대해서 반드시 주제:, 요약:, 가능한 질문: 이 세가지를 순차적으로 작성해야 합니다.
    2. 주제:는 입력 문서의 주제를 한 줄로 요약합니다.
    3. 요약:은 입력 문서를 5줄로 요약합니다.
    4. 가능한 질문:은 입력 문서로부부터 사람들이 할 수 있는 질문 세 가지를 파이썬 리스트 형태로 작성합니다.
    5. 가능한 질문:이 반드시 ["질문1", "질문2", "질문3"]과 같이 파이썬 리스트 형태로 작성되어야 하는 점에 유의하십시오.
    이제 시작합니다."""
    
    user_content = "입력: " + input_text + '\n답변:'
    
    print('\n최종 유저 프롬프트')
    print('==' * 50)
    print(user_content)
    print('==' * 50)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content

# 테스트 데이터 1: 로봇 관련 기사
test_text_1 = '''
새처럼 움직이는 항공 로봇 '레이븐'.
스위스 로잔 연방 공대(EPFL) 신원동 박사팀이 개발했는데요.
연구팀은 새의 엉덩이와 발목, 발을 참고해 뒷다리를 만들었습니다.
실제 새처럼 땅에서 걷고, 틈새를 건너뛰고, 위로 뛰어오릅니다.
껑충 뛰면서 날아오르는 이 기능이 핵심인데요.
기존 고정익 항공기처럼 지상을 달리지 않고 제자리에서 날아오릅니다.
연구팀은 레이븐의 점프 이륙이 점프하지 않고 날아오를 때보다 에너지 효율이 더 높다고 설명했는데요.
또 기존 고정익 항공기에 이 기능을 도입하면 복잡한 지형에서도 효율적으로 쓸 수 있을 거라고 덧붙였습니다.
#새 #로봇 #항공 #공대 #스위스 #신원동 #박사 #점프
'''

result = return_answer_analysis(input_text=test_text_1)
print(result)


# 테스트 데이터 2: 핵융합 관련 기사
test_text_2 = '''
꿈의 에너지’라고 불리는 핵융합 에너지가 5년 뒤 현실이 될지도 모른다. 마이크로소프트가 5년 뒤부터 핵융합으로 만든 전기를 사서 쓰겠다는 전기공급 계약을
마이크로소프트에게 전기를 공급할 곳은 미국 스타트업 ‘헬리온 에너지’다. 5월 10일 핵융합 발전 스타트업 헬리온 에너지가 마이크로소프트와 계약을 맺었다고
하지만 전문가들의 전망은 긍정적이지만은 않다. 사용할 수 있는 정도의 전기를 만들어 내는 실증 시험까지 거친 핵융합 연구가 아직 없기 때문이다. 대신 현실적
핵융합 발전 연구는 크게 공공 개발과 민간 개발, 둘로 나뉘는데 둘 중 핵융합 발전 목표 시기가 조금 더 빠른 민간 개발의 목표 시기가 2030년경이다. 다만 지난
'''
result = return_answer_analysis(input_text=test_text_2)
print(result)


# =============================================================================
# [5] 정형화된 응답 유도 - 키워드 추출 (Structured Output: Keywords)
# =============================================================================
# - Few-shot Prompting 기법 사용: 예시([Example])를 제공하여 원하는 출력 패턴을 학습시킴
# - 텍스트에서 핵심 키워드만 뽑아 파이썬 리스트 형태로 반환하도록 유도

def return_answer_keywords(input_text):
    system_prompt = """You are an expert at extracting keywords from a given sentence. Extract keywords from a given sentence that are key to the contex
    [Example]
    input: '외신에 따르면 중국 이카이글로벌 보도를 인용해 모더나가 중국 전용 mRNA 백신 개발을 위해 중국에 최대 10억달러(약 1조3017억원)를 투자하기로 했
    keyword: ['모더나', 'mRNA', '1조3017억원']
    Now the sentence you want to extract keywords from.
    The output format must be a list in Python.
    """
    user_content = "input: " + input_text + "\nkeyword: "
    
    print('\n최종 유저 프롬프트 (키워드 추출)')
    print('==' * 50)
    print(user_content)
    print('==' * 50)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0 # 창의성 최소화 (정확한 추출을 위해 0으로 설정)
    )

    return response.choices[0].message.content

# 테스트 데이터: 호텔 뷔페 가격 인상 뉴스
text_keywords = '''3일 업계에 따르면 서울 신라호텔의 더 파크뷰는 12월 1∼20일 저녁 가격을 19만5000원으로, 21∼31일 저녁 가격을 21만 5000원으로 각각 인상한다. 평
롯데호텔 서울의 뷔페 라세느는 12월 평일·주말 저녁 가격을 19만원으로 기존 대비 1만원 올린다. 크리스마스 연휴 때인 23∼25일과 연말 30∼31일 저녁 가격은 2
워커힐 호텔앤리조트는 그랜드 워커힐에서 운영하던 더뷔페를 비스타 워커힐로 확장 이전해 이날 가격을 올렸다. 평일·주말 저녁 가격은 15만9000원에서 18만9000
그랜드 인터컨티넨탈 서울 파르나스와 인터컨티넨탈 서울 코엑스도 다음주께 인상된 가격을 공지할 계획이다.'''

result = return_answer_keywords(input_text=text_keywords)
print(result)


# =============================================================================
# [6] 응용 - 제목 생성 (Title Generation)
# =============================================================================
# - 긴 텍스트를 입력받아 간결한 제목을 생성하는 Task

def return_answer_title(text):
    prompt = '''주어진 텍스트로부터 적절한 제목을 만들어줘. 제목은 간결하고 너무 길어서는 안 돼.'''
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

input_text_news = '''지난 11일 국회 정무위원회 전체 회의에선 이복현 금융감독원장의 해외 출장길이 도마 에 올랐다. 이 원장이 지난 8일부터 닷새간 동남아 싱가포
이 원장의 이번 출장은 국내 금융사의 투자 유치와 해외진출 확대를 지원하기 위해서다. 이 출장에는 윤종규 KB금융그룹 회장, 함영주 하나금융그룹 회장을 비롯해
야당의 지적처럼 금융권 일각에선 이번 출장을 두고 다소 의아하다는 시각이 있었다. 금감원이 해외에서 IR 행사를 하긴 하지만, 금융위원장이 아니라 금감원장의 해외 출장 논란
'''
result = return_answer_title(input_text_news)
print("\n--- [6] 제목 생성 결과 ---")
print(result)


# =============================================================================
# [7] 응용 - 감정 분석 (Sentiment Analysis)
# =============================================================================
# - 텍스트의 감정을 [positive, negative, neutral] 중 하나로 분류
# - 입력 텍스트 전처리(strip) 및 출력 결과 정리 포함

def return_answer_sentiment(text):
    prompt = '''
        주어진 텍스트가 긍정인지 중립인지 부정인지 예측하시오. 당신의 답변은 오로지 [positive, negative, neutral] 셋 중 하나여야만 합니다.
        ex) 'SK하이닉스가 2분기 실적에서 역대 최고를 기록하였다' => "positive"
    '''
    # 입력된 텍스트의 앞뒤 공백을 제거하여 프롬프트에 깔끔하게 전달
    clean_text = text.strip()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": clean_text + " => "}
        ]
    )
    # 응답값의 앞뒤 공백 제거 후 반환
    return response.choices[0].message.content.strip()

input_text_sentiment = '''
'씨씨에스 로고 씨씨에스 주가가 상한가다. 3일 기준 씨씨에스는 29.84%(825원) 상승한 3590원에 거래를 마쳤다. 상온 초전도체 이슈가 부각되면서
'''

result = return_answer_sentiment(input_text_sentiment)
print("\n--- [7] 감정 분석 결과 ---")
print("결과:", result)