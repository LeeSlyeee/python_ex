# =============================================================================
# [1] 라이브러리 임포트 및 환경 설정 (Library Import & Environment Setup)
# =============================================================================
import os  # 운영체제 상호작용 (파일 경로, 환경변수 접근 등)
import warnings  # 실행 시 발생하는 경고 메시지 제어
warnings.filterwarnings('ignore')  # 불필요한 Deprecation 경고 등을 숨김 처리

from dotenv import load_dotenv  # .env 파일의 환경변수 로드
from langchain_community.document_loaders import PyPDFLoader  # PDF 파일을 로드하여 정형 데이터로 변환하는 클래스
from langchain_classic.vectorstores import FAISS  # 페이스북에서 만든 효율적인 벡터 저장소 (CPU 기반 검색 엔진)
from langchain_classic.embeddings import OpenAIEmbeddings  # OpenAI 모델을 이용한 텍스트 임베딩 생성 도구
from langchain_classic.embeddings import HuggingFaceEmbeddings  # HuggingFace의 무료 오픈소스 임베딩 모델 사용 도구
from langchain_classic.chat_models import ChatOpenAI  # OpenAI의 챗 모델 (GPT-4o 등) 연동 클래스
from langchain_classic.chains import RetrievalQA  # 질문과 답변(Q&A) 기능을 하나로 묶어주는 체인 기술

# =============================================================================
# [2] 환경 변수 및 경로 설정 (Load Environment Variables & Path)
# =============================================================================
# .env 파일에 저장된 API 키 값을 시스템 환경변수로 로드합니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 현재 실행 중인 파이썬 스크립트의 절대 경로를 가져옵니다.
script_dir = os.path.dirname(os.path.abspath(__file__))
# 데이터 폴더 안의 PDF 파일까지의 전체 경로를 생성합니다.
pdf_path = os.path.join(script_dir, "data", "The_Adventures_of_Tom_Sawyer.pdf")

# =============================================================================
# [3] 문서 로딩 (Document Loading)
# =============================================================================
# 1. PyPDFLoader 객체 생성: 지정된 경로의 PDF 파일을 읽을 준비를 합니다.
loader = PyPDFLoader(pdf_path)

# 2. loader.load(): PDF의 각 페이지를 텍스트 데이터와 메타데이터가 담긴 'Document' 객체 리스트로 변환합니다.
document = loader.load()

# 3. 데이터 확인: 로드된 문서 중 6번째 페이지(인덱스 5)의 내용을 최대 5000자까지 확인합니다.
print("\n--- [3] 문서 로딩 결과 확인 (6번째 페이지) ---")
print(document[5].page_content[:5000])

# =============================================================================
# [4] OpenAI 임베딩 및 벡터 저장소 구축 (OpenAI Embeddings & Vector Store)
# =============================================================================
# 1. OpenAI 임베딩 모델 초기화: 텍스트를 고차원 숫자 벡터로 변환하는 엔진입니다.
embeddings_openai = OpenAIEmbeddings() 

# 2. FAISS 벡터 저장소 생성:
# document 리스트를 임베딩 모델을 통해 벡터화하고, FAISS 데이터베이스에 저장합니다.
# 이를 통해 나중에 사용자의 질문과 가장 유사한 문서 조각을 빠르게 검색할 수 있게 됩니다.
db = FAISS.from_documents(document, embeddings_openai)

# 3. 임베딩 벡터 확인 (테스트용):
# "진희는 강아지를 키우고 있습니다..."라는 텍스트가 어떤 숫자로 변환되는지 상위 50개 값만 출력합니다.
print("\n--- [4] OpenAI 임베딩 벡터 샘플 (50개) ---")
text = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"
text_embedding_openai = embeddings_openai.embed_query(text)
print(text_embedding_openai[:50])

# =============================================================================
# [5] 타사 임베딩 모델(HuggingFace) 테스트 (Alternative Embeddings)
# =============================================================================
# 1. HuggingFace 임베딩 초기화: 
# "sentence-transformers/all-MiniLM-L6-v2" 모델을 사용하여 무료로 임베딩을 수행합니다.
embeddings_huggingface = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# 2. 동일한 텍스트로 HuggingFace 방식의 임베딩 벡터 생성 및 확인
print("\n--- [5] HuggingFace 임베딩 벡터 샘플 (50개) ---")
text_embedding_huggingface = embeddings_huggingface.embed_query(text)
print(text_embedding_huggingface[:50])

# =============================================================================
# [6] RAG(Retrieval-Augmented Generation) 체인 구성 (QA Chain)
# =============================================================================
# 1. LLM 초기화: 답변을 생성할 똑똑한 두뇌(GPT-4o-mini)를 준비합니다.
# temperature=0: 가장 확률적으로 높은, 일관된 답변을 하도록 설정합니다.
llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')

# 2. 검색기(Retriever) 설정:
# 앞서 구축한 FAISS DB에서 사용자의 질문과 관련된 정보를 찾아오는 역할을 부여합니다.
retriever = db.as_retriever()

# 3. RetrievalQA 체인 생성:
# llm(두뇌) + retriever(자료실)를 연결합니다.
# chain_type="stuff": 검색된 문서 조각들을 모두 하나로 합쳐서 LLM에게 전달하는 가장 일반적인 방식입니다.
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# =============================================================================
# [7] 최종 질문 및 응답 (Query & Final Result)
# =============================================================================
# 사용자 질문 정의 (PDF 내용 기반)
query = "마을 무덤에 있던 남자를 죽인 사람은 누구니?"

# qa 체인 가동: 
# [과정: 질문 접수 -> DB에서 관련 내용 검색 -> 검색된 내용+질문을 LLM에게 전달 -> LLM이 최종 답변 생성]
# invoke(query) 가 권장되나 기존 코드 형식인 ({'query': query})를 유지합니다.
result = qa({"query": query})

# 결과 출력
print("\n--- [7] 최종 Q&A 결과 ---")
print(f"질문: {query}")
print(f"답변: {result['result']}")