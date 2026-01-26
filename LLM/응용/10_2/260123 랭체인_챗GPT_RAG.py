# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os
from dotenv import load_dotenv

# 문서 로더(Document Loader): 텍스트 파일을 읽어오는 도구
from langchain_classic.document_loaders import TextLoader

# 텍스트 분할기(Text Splitter): 긴 텍스트를 작은 청크(Chunk)로 나누는 도구
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# 임베딩 모델(Embedding Model): 텍스트를 벡터(숫자 리스트)로 변환하는 모델
from langchain_openai import OpenAIEmbeddings

# 벡터 저장소(Vector Store): 벡터화된 데이터를 저장하고 검색하는 데이터베이스 (Chroma 사용)
from langchain_classic.vectorstores import Chroma

# 채팅 모델(Chat Model): LLM(Large Language Model) 래퍼 클래스
from langchain_classic.chat_models import ChatOpenAI

# 질의응답 체인(QA Chain): 검색된 문서와 질문을 LLM에 전달하여 답을 얻는 체인 로드 함수
from langchain_classic.chains.question_answering import load_qa_chain


# .env 파일에서 환경 변수 로드 (API Key 등 보안 정보 관리)
# override=True: 시스템 환경 변수보다 .env 파일의 값을 우선함
load_dotenv(override=True)

# OpenAI API Key를 환경 변수에서 가져옴
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =============================================================================
# [2] 문서 로드 (Document Loading)
# =============================================================================
# 현재 스크립트 파일의 디렉토리 경로를 구합니다. (절대 경로 사용을 위함)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 로드할 대상 파일("AI.txt")의 절대 경로를 생성합니다.
file_path = os.path.join(current_dir, "AI.txt")

# TextLoader를 사용하여 파일을 로드합니다.
# 반환값은 Document 객체의 리스트입니다. (page_content와 metadata를 포함)
documents = TextLoader(file_path).load()


# =============================================================================
# [3] 텍스트 분할 (Text Splitting)
# =============================================================================
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    문서를 지정된 크기의 청크(Chunk)로 분할합니다.
    
    Args:
        documents: 분할할 문서 리스트
        chunk_size: 각 청크의 최대 글자 수 (기본값: 1000)
        chunk_overlap: 청크 간 중복되는 글자 수 (문맥 유지를 위해 사용, 기본값: 20)
        
    Returns:
        docs: 분할된 문서(Chunk) 리스트
    """
    # RecursiveCharacterTextSplitter 초기화
    # 문단 -> 문장 -> 단어 순으로 재귀적으로 텍스트를 나눔
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 텍스트 분할 수행
    docs = text_splitter.split_documents(documents)
    return docs

# 함수 호출하여 문서 분할 실행
docs = split_docs(documents)


# =============================================================================
# [4] 임베딩 및 벡터 저장소 생성 (Embedding & VectorStore)
# =============================================================================
# OpenAI의 "text-embedding-ada-002" 모델을 사용하여 텍스트를 벡터로 변환하는 객체 생성
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

# 분할된 문서(docs)를 벡터로 변환하여 Chroma DB에 저장
# persist_directory="data": 생성된 벡터 DB를 로컬 디스크의 "data" 폴더에 저장 (영속화)
db = Chroma.from_documents(docs, embeddings, persist_directory="data")


# =============================================================================
# [5] LLM 및 QA 체인 초기화 (LLM & QA Chain Initialization)
# =============================================================================
# 사용할 LLM 모델 이름 설정 (참고: gpt-4.1-mini는 오타일 가능성이 높음 -> gpt-4o-mini 등으로 수정 권장되나 원본 유지)
model_name = "gpt-4.1-mini"

# ChatOpenAI 클래스로 LLM 객체 생성
llm = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY)

# 질의응답(Question Answering)을 위한 체인 로드
# chain_type="stuff": 검색된 모든 문서를 프롬프트 하나에 다 채워넣는(Stuff) 방식
# verbose=True: 체인 실행 과정을 상세히 출력
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)


# =============================================================================
# [6] 질의응답 실행 (Execution)
# =============================================================================
# 질문 정의
query = "AI란?"

# 1. 유사도 검색 (Similarity Search)
# 질문(query)과 가장 유사한 내용을 담고 있는 문서를 벡터 DB에서 검색
matching_docs = db.similarity_search(query)

# 2. 체인 실행 (Run Chain)
# 검색된 문서(input_documents)와 질문(question)을 체인에 전달하여 답변 생성
# run() 메서드로 체인 실행 (deprecated 경고가 뜰 수 있으나 원본 코드 유지)
answer = chain.run(input_documents=matching_docs, question=query)

# 최종 답변 출력
print(answer)