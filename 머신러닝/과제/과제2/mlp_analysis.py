# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 데이터 분석 및 머신러닝 모델링에 필요한 라이브러리를 불러옵니다.

import pandas as pd # 데이터프레임 처리를 위한 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import sys          # 시스템 관련 기능 (현재 미사용이나 임포트 유지)

# Scikit-learn 관련 모듈 임포트
# 데이터 분할: 학습용(Train)과 테스트용(Test) 데이터로 나누기
from sklearn.model_selection import train_test_split
# 데이터 전처리: 스케일링(StandardScaler) 및 라벨 인코딩(LabelEncoder)
from sklearn.preprocessing import StandardScaler, LabelEncoder
# 모델: 다층 퍼셉트론(Multi-layer Perceptron) 분류기
from sklearn.neural_network import MLPClassifier
# 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt # 시각화 라이브러리
import seaborn as sns           # 시각화 라이브러리
import os                       # 파일 경로 처리

# 평가 지표: 분류 리포트, 오차 행렬, 정확도 점수, ROC AUC, ROC 곡선
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# 시각화 설정: 한글 폰트 (Mac: AppleGothic, Win: Malgun Gothic)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False 

def analyze_student_performance(file_path):
    # 결과 저장 경로 설정 (데이터 파일과 동일한 위치)
    save_dir = os.path.dirname(file_path)

    # =============================================================================
    # [2] 데이터 로드 (Data Loading)
    # =============================================================================
    print(f"Loading data from {file_path}...")
    try:
        # CSV 파일 로드
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # =============================================================================
    # [3] 데이터 전처리 (Data Preprocessing)
    # =============================================================================
    print("\n[Data Preprocessing]")
    
    # 1. 범주형 변수 처리 (Categorical Features)
    # 타겟 변수('Class')를 제외한 범주형(object 타입) 컬럼 식별
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Class') 
    
    print(f"Categorical columns to encode: {categorical_cols}")
    
    # 특성에 대한 원-핫 인코딩 (One-Hot Encoding)
    # 범주형 데이터를 기계학습 모델이 이해할 수 있도록 0과 1로 변환
    X = pd.get_dummies(df.drop('Class', axis=1), columns=categorical_cols, drop_first=True)
    
    # 2. 타겟 변수 인코딩 (Target Encoding)
    # 타겟 변수 'Class'를 숫자로 변환 (예: L->1, M->2, H->0 등)
    le = LabelEncoder()
    y = le.fit_transform(df['Class'])
    # 변환된 클래스 정보 출력
    print(f"Target classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 3. 데이터 분할 (Data Split)
    # 전체 데이터를 학습용(80%)과 테스트용(20%)으로 분리
    # random_state=42: 재현성을 위한 난수 시드 고정
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. 스케일링 (Scaling)
    # 데이터의 스케일(범위)을 맞추기 위해 표준화(StandardScaler) 적용
    # 평균 0, 표준편차 1이 되도록 변환 (MLP는 스케일에 민감함)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # 학습 데이터로 스케일러 학습 및 변환
    X_test_scaled = scaler.transform(X_test)       # 테스트 데이터는 학습된 스케일러로 변환만 수행

    # =============================================================================
    # [4] 모델 학습 (Model Training) & 하이퍼파라미터 튜닝
    # =============================================================================
    print("\n[Model Training & Hyperparameter Tuning]")
    
    # MLP 분류기 객체 생성
    mlp = MLPClassifier(max_iter=1000, random_state=42)

    # 튜닝할 파라미터 그리드 정의
    # 튜닝할 파라미터 그리드 정의
    # hidden_layer_sizes: 은닉층의 개수와 뉴런 수를 다양하게 조합
    # learning_rate_init: 초기 학습률을 조정하여 학습 속도 제어
    # alpha: L2 규제 파라미터 (과적합 방지)
    # activation: 활성화 함수 (ReLU, Hyperbolic Tangent)
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128)],
        'learning_rate_init': [0.001, 0.01, 0.0001],
        'alpha': [0.0001, 0.001, 0.01],
        'activation': ['relu', 'tanh']
    }

    # GridSearchCV 설정 (5-Fold 교차 검증)
    # cv=5: 데이터를 5개로 나누어 교차 검증 수행
    # n_jobs=-1: 모든 CPU 코어를 사용하여 병렬 처리 (속도 향상)
    # verbose=1: 튜닝 과정 중 로그 출력 활성화
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    
    print("Hyperparameter tuning started... (This may take a while)")
    # 그리드 서치 실행 (모든 조합에 대해 모델 학습 및 평가)
    grid_search.fit(X_train_scaled, y_train)
    
    # 최적의 파라미터와 점수 출력
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # 최적의 모델로 선택
    best_mlp = grid_search.best_estimator_

    # =============================================================================
    # [5] 예측 및 평가 (Prediction & Evaluation)
    # =============================================================================
    print("\n[Evaluation of Best Model]")
    # 튜닝된 최적의 모델로 테스트 데이터 예측
    y_pred = best_mlp.predict(X_test_scaled)
    
    # 정확도(Accuracy) 계산 및 출력
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # 분류 리포트 출력 (정밀도, 재현율, F1 점수 등)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 오차 행렬(Confusion Matrix) 출력
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # =============================================================================
    # [6] 결과 시각화 (Visualization)
    # =============================================================================
    # 튜닝된 모델의 성능을 시각적으로 분석합니다.
    print("\n[Visualization]")
    
    # 1. Confusion Matrix 히트맵
    # 실제값과 예측값의 관계를 색상으로 표현하여 오분류 패턴 파악
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_)
    plt.title('MLP Confusion Matrix (Best Model)')
    plt.xlabel('Predicted (예측값)')
    plt.ylabel('Actual (실제값)')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'mlp_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to {cm_path}")

    # 2. ROC Curve (Multi-class OvR)
    # 다중 클래스 분류이므로 One-vs-Rest 방식으로 각 클래스별 ROC 곡선 시각화
    plt.figure(figsize=(10, 8))
    # y_test를 One-Hot 형태로 변환 (ROC 커브 그리기 위해 필요)
    y_test_bin = pd.get_dummies(y_test) 
    # 또는 predict_proba 결과 활용을 위해 y_test를 이진화 (Binarize)
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=sorted(np.unique(y)))
    n_classes = y_test_bin.shape[1]
    
    # 각 클래스에 속할 확률 예측
    y_score = best_mlp.predict_proba(X_test_scaled)

    # 클래스별 ROC 곡선 그리기
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
        class_name = le.classes_[i]
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'ROC curve ({class_name}) (area = {roc_auc:.2f})')

    # 무작위 추측 선 (Random Guess) 표시
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (허위 양성 비율)')
    plt.ylabel('True Positive Rate (진짜 양성 비율)')
    plt.title('Multi-class ROC Curve (Best MLP)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(save_dir, 'mlp_roc_curve.png')
    plt.savefig(roc_path)
    print(f"ROC Curve saved to {roc_path}")

    # 3. Loss Curve (학습 곡선)
    # 학습 반복 횟수(Iterations)에 따른 손실값(Loss) 변화 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(best_mlp.loss_curve_)
    plt.title('Best MLP Training Loss Curve (학습 곡선)')
    plt.xlabel('Iterations (반복 횟수)')
    plt.ylabel('Loss (손실)')
    plt.grid(True)
    loss_path = os.path.join(save_dir, 'mlp_loss_curve.png')
    plt.savefig(loss_path)
    print(f"Loss Curve saved to {loss_path}")


if __name__ == "__main__":
    # 데이터 파일 경로 설정
    data_path = "/Volumes/Crucial 2TB/sesac_study/머신러닝/과제/과제2/xAPI-Edu-Data.csv"
    # 분석 함수 실행
    analyze_student_performance(data_path)
