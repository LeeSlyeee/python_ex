# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
import cv2 as cv   # OpenCV 라이브러리 (컴퓨터 비전 이미지/비디오 처리)
import numpy as np # 수치 연산 및 배열 처리를 위한 NumPy
import sys         # 시스템 관련 기능 (종료 등)
import os          # 파일 경로 및 운영체제 상호작용

# =============================================================================
# [2] 이미지 로드 및 기본 표시 (Image Loading & Display)
# =============================================================================
# 현재 스크립트 파일이 위치한 디렉토리 경로 가져오기
script_dir = os.path.dirname(__file__)

# 이미지 파일 경로 설정 및 읽기 (imread)
# cv.imread: 이미지를 로드하여 NumPy 배열로 반환
# os.path.join: 운영체제에 맞는 경로 구분자를 사용하여 경로 생성
img = cv.imread(os.path.join(script_dir, 'soccer.jpg'))

# 이미지가 정상적으로 로드되었는지 확인
if img is None:
    # 이미지가 없으면 오류 메시지 출력 후 프로그램 종료
    sys.exit('파일을 찾을 수 없습니다.')

# 윈도우 창에 이미지 표시 (imshow)
# 첫 번째 인자: 윈도우 창 이름
# 두 번째 인자: 표시할 이미지 변수
cv.imshow('Image Display', img)

# 키보드 입력 대기 (waitKey)
# 인자가 없거나 0이면 무한 대기, 키 입력 시 해당 키의 ASCII 코드 반환 후 진행
print(cv.waitKey()) 

# 생성된 모든 OpenCV 윈도우 창 닫기
cv.destroyAllWindows()


# =============================================================================
# [3] 이미지 변환 및 저장 (Image Conversion, Resizing & Saving)
# =============================================================================
# 원본 이미지 다시 로드
img = cv.imread(os.path.join(script_dir, 'soccer.jpg'))

# 예외 처리: 이미지 로드 실패 시 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 컬러 이미지를 흑백(Grayscale) 이미지로 변환
# cv.cvtColor: 색상 공간 변환 (BGR -> Gray)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 이미지 크기 조절 (Resizing)
# fx=0.5, fy=0.5: 가로, 세로 크기를 0.5배(반)로 축소
# dsize=(0,0): 절대 크기를 지정하지 않고 비율(fx, fy)로 크기 조절
gray_small=cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)

# 변환된 이미지 파일로 저장 (imwrite)
# 'soccer_gray.jpg': 저장할 파일명
# gray: 저장할 이미지 데이터
cv.imwrite('soccer_gray.jpg', gray)
cv.imwrite('soccer_gray_small.jpg', gray_small)

# 결과 이미지들을 윈도우 창에 표시
cv.imshow('Color Image', img)           # 원본 컬러 이미지
cv.imshow('Gray Image', gray)           # 흑백 이미지
cv.imshow('Gray Image Small', gray_small) # 축소된 흑백 이미지

# 키 입력 대기 후 모든 창 닫기
cv.waitKey()
cv.destroyAllWindows()


# =============================================================================
# [4] 웹캠 영상 출력 (Webcam Video Display)
# =============================================================================
# 웹캠 장치 연결 (cv.VideoCapture)
# 0: 기본 카메라 장치 인덱스
cap=cv.VideoCapture(0)

# 카메라 연결 상태 확인
if not cap.isOpened():
    sys.exit('카메라 연결 실패') # 연결 실패 시 종료

# 무한 루프를 통해 비디오 프레임 연속 처리
while True:
    # 카메라로부터 프레임 읽기 (cap.read)
    # ret: 성공 여부 (True/False)
    # frame: 캡처된 이미지 프레임
    ret, frame=cap.read()
    
    # 프레임을 읽지 못했을 경우 (카메라 오류 또는 영상 종료)
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
        
    # 읽어온 프레임을 윈도우 창에 표시
    cv.imshow('Video Display', frame)

    # 1ms 대기하며 키 입력 확인
    key=cv.waitKey(1)
    
    # 'q' 키(quit)를 누르면 루프 종료
    if key==ord('q'):
        break

# 카메라 자원 해제
cap.release()
# 윈도우 창 닫기
cv.destroyAllWindows()


# =============================================================================
# [5] 영상 캡처 및 병합 (Video Capture & Concatenation)
# =============================================================================
# 웹캠 연결
cap=cv.VideoCapture(0)

# 연결 확인
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

# 캡처한 프레임을 저장할 리스트 초기화
frames = []

while True:
    # 프레임 읽기
    ret, frame=cap.read()
    
    # 읽기 실패 시 중단
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
        
    # 실시간 영상 표시
    cv.imshow('Video Display', frame)
  
    # 키 입력 처리
    key=cv.waitKey(1)
    
    # 'c' 키를 누르면 현재 프레임을 리스트에 저장 (Capture)
    if key==ord('c'):
        frames.append(frame)
    # 'q' 키를 누르면 종료 (Quit)
    elif key==ord('q'):
        break

# 카메라 자원 해제
cap.release()
# 영상 윈도우 닫기
cv.destroyAllWindows()

# 캡처된 프레임이 있는 경우 처리
if len(frames) > 0:
    # 첫 번째 프레임을 기준으로 설정
    imgs=frames[0]
    
    # 최대 3장까지 가로로 이어 붙이기 (np.hstack)
    # min(3, len(frames)): 캡처된 장수가 3장 미만일 경우 처리
    for i in range(1, min(3, len(frames))):
        imgs=np.hstack((imgs, frames[i])) # Horizontal Stack (가로 병합)

    # 수집된(병합된) 프레임 표시
    cv.imshow('collected Frames', imgs)

    # 키 입력 시까지 대기 후 종료
    cv.waitKey()
    cv.destroyAllWindows()


# =============================================================================
# [6] 이미지 그리기 - 도형 및 텍스트 (Drawing Shapes & Text)
# =============================================================================
# 이미지 로드
img=cv.imread(os.path.join(script_dir, 'girl_laughing.jpg'))

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 직사각형 그리기 (cv.rectangle)
# img: 그릴 대상 이미지
# (830,30): 시작점 좌표 (x, y)
# (1000,200): 종료점 좌표 (x, y)
# (0,0,255): 색상 (BGR 순서 -> Red)
# 2: 선 두께
cv.rectangle(img, (830,30),(1000,200),(0,0,255),2)

# 텍스트 쓰기 (cv.putText)
# img: 텍스트를 넣을 이미지
# 'laugh': 표시할 텍스트
# (830,24): 텍스트 시작 위치 좌표
# cv.FONT_HERSHEY_SIMPLEX: 폰트 종류
# 1: 폰트 크기 스케일
# (255,0,0): 텍스트 색상 (BGR -> Blue)
# 2: 텍스트 두께
print(cv.putText(img, 'laugh', (830,24), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2))

# 결과 이미지 표시
cv.imshow('Draw', img)
print(cv.waitKey())
cv.destroyAllWindows()


# =============================================================================
# [7] 마우스 이벤트 1 - 클릭 시 도형 그리기 (Mouse Event - Click to Draw)
# =============================================================================
# 이미지 로드
img=cv.imread(os.path.join(script_dir, 'girl_laughing.jpg'))

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 마우스 콜백 함수 정의 (이벤트 핸들러)
# event: 마우스 이벤트 종류 (클릭, 이동 등)
# x, y: 이벤트 발생 좌표
def draw(event,x,y,flags,param):
    # 왼쪽 버튼 클릭 시 (LBUTTONDOWN) -> 빨간색 사각형
    if event==cv.EVENT_LBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+200,y+200),(0,0,255),2)
    # 오른쪽 버튼 클릭 시 (RBUTTONDOWN) -> 파란색 사각형
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+100,y+100),(255,0,0),2)
    
    # 그림이 그려진 이미지를 갱신하여 표시
    cv.imshow('Drawing',img)

# 윈도우 생성 및 이미지 표시
cv.namedWindow('Drawing')
cv.imshow('Drawing',img)

# 윈도우와 콜백 함수 연결
# 'Drawing' 윈도우에서 마우스 이벤트 발생 시 'draw' 함수 호출
cv.setMouseCallback('Drawing',draw)

# 'q' 키를 누를 때까지 루프 유지
while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break


# =============================================================================
# [8] 마우스 이벤트 2 - 드래그로 그리기 (Mouse Event - Drag to Draw)
# =============================================================================
# 이미지 로드
img=cv.imread(os.path.join(script_dir, 'girl_laughing.jpg'))

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 드래그 시작 좌표 저장을 위한 전역 변수
# ix, iy: Initial X, Initial Y
# 함수 내부에서 값을 변경하기 위해 global 키워드 사용 예정
def draw(event,x,y,flags,param):
    global ix,iy # 전역 변수 참조
    
    # 왼쪽 버튼을 누른 시점 (시작)
    if event==cv.EVENT_LBUTTONDOWN:
        ix,iy=x,y # 시작 좌표 저장
        
    # 왼쪽 버튼을 뗀 시점 (종료)
    elif event==cv.EVENT_LBUTTONUP:
        # 시작 좌표(ix, iy)에서 현재 좌표(x, y)까지 빨간색 사각형 그리기
        cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2)
        
    # 변경된 이미지 표시
    cv.imshow('Drawing',img)

cv.namedWindow('Drawing')
cv.imshow('Drawing',img)
cv.setMouseCallback('Drawing',draw)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break


# =============================================================================
# [9] 마우스 이벤트 3 - 페인팅 브러쉬 (Painting Brush)
# =============================================================================
# 이미지 로드 (배경으로 사용)
img = cv.imread(os.path.join(script_dir, 'soccer.jpg'))

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 브러쉬 설정
BrushSiz=5 # 붓 크기 (반지름)
LColor,RColor=(255,0,0),(0,0,255) # LClick: 파랑, RClick: 빨강 (BGR)

# 페인팅 콜백 함수
def painting(event,x,y,flags,param):
    # 왼쪽 버튼 클릭 시 점 찍기
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,LColor,-1) # -1: 내부 채움
        
    # 오른쪽 버튼 클릭 시 점 찍기
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,RColor,-1)
        
    # 마우스를 움직이면서(MOUSEMOVE) 동시에 왼쪽 버튼이 눌려있는 상태(FLAG_LBUTTON)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)
        
    # 마우스를 움직이면서(MOUSEMOVE) 동시에 오른쪽 버튼이 눌려있는 상태(FLAG_RBUTTON)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img,(x,y),BrushSiz,RColor,-1)
    
    # 그려진 이미지 갱신
    cv.imshow('Painting',img)


cv.namedWindow('Painting')
cv.imshow('Painting',img)

# 콜백 연결
cv.setMouseCallback('Painting',painting)

# 종료 루프
while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
