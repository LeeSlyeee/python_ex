# =============================================================================
# [라이브러리 임포트]
# =============================================================================
import os
import json
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =============================================================================
# [1] 설정 및 모델 로드
# =============================================================================

# 현재 스크립트 파일이 위치한 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

# 학습된 '미세 조정' 모델 파일 경로 (callbacks으로 저장된 best model)
model_path = os.path.join(script_dir, 'best_garbage_model.h5')

# 클래스 정보가 담긴 JSON 파일 경로
json_path = os.path.join(script_dir, 'garbage_classes.json')

# 모델 파일 확인
if not os.path.exists(model_path):
    print(f"Error: 모델 파일이 없습니다. ({model_path})")
    print("먼저 전이 학습 스크립트를 실행하여 모델을 생성해주세요.")
    exit()

# 필요한 Keras 레이어 및 모델 임포트
import tensorflow as tf
from tensorflow.keras.layers import Dense

# [Fix] Keras 버전 호환성 문제 해결을 위한 커스텀 Dense 클래스
# 저장된 모델에 'quantization_config'가 포함되어 로드 시 에러가 발생하는 경우 이를 제거함
class FixedDense(Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

# 모델 로드
print("미세 조정된 모델 로드 중...", end="")

# 1. 클래스 이름 로드
with open(json_path, 'r') as f:
    class_names = json.load(f)

# 2. 모델 로드 (custom_objects 사용)
try:
    # compile=False를 권장 (예측 전용)
    model = tf.keras.models.load_model(model_path, 
                                       custom_objects={'Dense': FixedDense},
                                       compile=False)
    print("완료!")
except Exception as e:
    print(f"\n[오류] 모델 로드 다시 실패: {e}")
    print("모델 파일을 다시 확인하거나 training 스크립트의 설정을 점검하세요.")
    exit()

# =============================================================================
# [2] 예측 함수 정의
# =============================================================================
def predict_image(image_path):
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"Error: 이미지를 찾을 수 없습니다. ({image_path})")
        return

    try:
        # 이미지 읽기 (한글 경로 대응)
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"이미지 로딩 실패: {e}")
        return

    if img is None:
        print("이미지 디코딩 실패")
        return

    # 전처리 1: 리사이징 (224x224)
    img_resized = cv2.resize(img, (224, 224))
    
    # 전처리 2: BGR -> RGB 변환 (중요!)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 전처리 3: 차원 확장 및 스케일링 (1, 224, 224, 3)
    input_data = np.expand_dims(img_rgb, axis=0)
    input_data = input_data.astype('float32') # 형변환
    input_data = preprocess_input(input_data) # -1 ~ 1 정규화

    # 예측 수행
    predictions = model.predict(input_data)
    
    # 결과 해석
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # 결과 출력
    print("\n" + "="*40)
    print(f"모델: Fine-Tuned MobileNetV2")
    print(f"대상 이미지: {os.path.basename(image_path)}")
    print(f"예측 결과: {predicted_class}")
    print(f"확신의 정도(Confidence): {confidence:.2f}%")
    print("="*40 + "\n")

    # 결과 시각화 및 저장
    plt.figure(figsize=(6, 6))
    img_display_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display_rgb)
    
    # 그래프 제목에 결과 표시
    plt.title(f"Fine-Tuned Prediction: {predicted_class}\n({confidence:.2f}%)", fontsize=14, color='blue')
    plt.axis('off')
    
    # 이미지 저장
    save_path = os.path.join(script_dir, 'prediction_result_finetuned.png')
    plt.savefig(save_path)
    print(f"결과 이미지가 저장되었습니다: {save_path}")

# =============================================================================
# [3] 실행
# =============================================================================
if __name__ == "__main__":
    target_image = os.path.join(script_dir, 'plastic.jpg')
    predict_image(target_image)
