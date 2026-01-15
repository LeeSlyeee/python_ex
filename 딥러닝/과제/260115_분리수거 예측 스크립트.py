# =============================================================================
# [라이브러리 임포트]
# =============================================================================
import os                           # 운영체제 명령(파일 경로 등) 사용을 위한 모듈
import json                         # JSON 데이터 파싱을 위한 모듈
import numpy as np                  # 수치 연산 및 배열 처리를 위한 라이브러리
import cv2                          # OpenCV 라이브러리 (이미지 읽기 및 처리)
import tensorflow as tf             # 딥러닝 프레임워크 텐서플로우
import matplotlib.pyplot as plt     # 이미지 시각화 도구
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # MobileNetV2 전용 전처리 함수

# =============================================================================
# [1] 설정 및 모델 로드 (Settings & Load Model)
# =============================================================================

# 현재 스크립트 파일이 위치한 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

# 학습된 모델 파일(.h5)의 경로 지정 (전이 학습된 모델 사용)
model_path = os.path.join(script_dir, 'garbage_classification_model_transfer.h5')

# 클래스 정보가 담긴 JSON 파일 경로 지정
json_path = os.path.join(script_dir, 'garbage_classes.json')

# 모델 파일 존재 여부 확인
if not os.path.exists(model_path):
    print(f"Error: 모델 파일이 없습니다. ({model_path})")
    print("전이 학습 스크립트(260115_garbage_classification_transfer.py)가 완료되었는지 확인해주세요.")
    exit() # 파일이 없으면 프로그램 종료

# 저장된 Keras 모델 로드
print("모델 로드 중...")
model = tf.keras.models.load_model(model_path)
print("모델 로드 완료!")

# JSON 파일에서 클래스 이름 목록 읽어오기
# 예: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
with open(json_path, 'r') as f:
    class_names = json.load(f)

# 로드된 클래스 정보 출력
print(f"분류 가능한 클래스: {class_names}")


# =============================================================================
# [2] 예측 함수 정의 (Prediction Function)
# =============================================================================
def predict_image(image_path):
    """
    주어진 이미지 경로의 이미지를 읽어 전처리 후 모델 예측 수행
    """
    
    # 이미지 파일 존재 여부 확인
    if not os.path.exists(image_path):
        print(f"Error: 이미지를 찾을 수 없습니다. ({image_path})")
        return

    # 1. 이미지 읽기
    # 일반 cv2.imread는 한글 경로 인식에 문제가 있을 수 있어 numpy 버퍼로 읽는 방식 사용 권장
    try:
        # 파일을 바이너리 형태로 읽어서 numpy 배열로 변환
        img_array = np.fromfile(image_path, np.uint8)
        # 이미지 디코딩 (BGR 형식으로 로드됨에 주의)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"이미지 읽기 오류: {e}")
        return
    
    # 이미지가 정상적으로 읽혔는지 확인
    if img is None:
        print("이미지를 열 수 없습니다. (파일 손상 또는 경로 문제)")
        return

    # 2. 이미지 전처리
    # (1) 리사이징: 모델이 학습될 때 사용한 입력 크기(224x224)로 변경
    # 주의: cv2.resize의 인자는 (너비, 높이) 순서
    img_resized = cv2.resize(img, (224, 224))

    # (2) 색상 공간 변환: OpenCV는 BGR을 쓰지만, 학습된 모델(MobileNetV2)은 RGB를 기대함
    # 이 단계를 건너뛰면 색상 정보가 뒤바뀌어 예측 성능이 크게 떨어짐
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # (3) 차원 확장: 모델은 배치 입력을 기대하므로 앞에 차원을 하나 추가
    # (224, 224, 3) -> (1, 224, 224, 3)
    input_data = np.expand_dims(img_rgb, axis=0)
    
    # (4) 스케일링: MobileNetV2 전용 전처리 함수 사용 (픽셀값을 [-1, 1] 범위로 조정 등)
    input_data = input_data.astype('float32') # 실수형으로 변환
    input_data = preprocess_input(input_data)

    # 3. 예측 수행
    # 모델에 전처리된 데이터를 넣어 각 클래스별 확률 예측
    predictions = model.predict(input_data)
    
    # 4. 결과 해석
    # 가장 높은 확률을 가진 클래스의 인덱스 추출
    predicted_index = np.argmax(predictions[0])
    # 인덱스를 클래스 이름으로 변환
    predicted_class = class_names[predicted_index]
    # 해당 클래스의 확률(신뢰도) 가져오기 (퍼센트로 변환)
    confidence = predictions[0][predicted_index] * 100

    # 결과 출력
    print("\n" + "="*30)
    print(f"모델: MobileNetV2 (Transfer Learning)")
    print(f"이미지 파일: {os.path.basename(image_path)}")
    print(f"예측 결과: {predicted_class}")
    print(f"확률: {confidence:.2f}%")
    print("="*30 + "\n")

    # (옵션) 결과 이미지 시각화
    # 원본 이미지를 BGR에서 RGB로 변환하여 Matplotlib으로 출력
    # (여기서는 전처리된 이미지가 아니라 원본 이미지를 보여줌)
    img_display_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_display_rgb)
    plt.title(f"Prediction: {predicted_class} ({confidence:.1f}%)")
    plt.axis('off') # 축 숨기기
    
    # 결과 이미지 저장 png 파일로 저장
    save_path = os.path.join(script_dir, 'prediction_result_transfer.png')
    plt.savefig(save_path)
    print(f"결과 이미지가 저장되었습니다: {save_path}")

# =============================================================================
# [3] 테스트 실행 (Run Test)
# =============================================================================
if __name__ == "__main__":
    # 테스트할 이미지 경로 설정
    # 기본값: 같은 디렉토리에 있는 'plastic.jpg' 파일
    target_image = os.path.join(script_dir, 'plastic.jpg')
    
    # 사용자 입력을 받으려면 아래 주석 해제 (경로 입력 가능)
    # target_image = input("테스트할 이미지 경로를 입력하세요: ").strip().replace("'", "").replace('"', "")
    
    # 예측 함수 호출
    predict_image(target_image)

