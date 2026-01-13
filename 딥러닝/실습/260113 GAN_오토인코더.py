# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
from tensorflow.keras.datasets import mnist # MNIST 손글씨 데이터셋
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout # 각종 레이어
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D # 활성화 및 정규화 레이어
from tensorflow.keras.layers import Conv2D, MaxPooling2D # CNN 관련 레이어
from tensorflow.keras.models import Sequential, Model # 모델 구성 클래스

import numpy as np              # 수치 연산 및 배열 처리
import matplotlib.pyplot as plt # 시각화 라이브러리


# =============================================================================
# [2] GAN 모델 설계 (GAN Model Design)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 생성자 (Generator) 모델 설계
# -----------------------------------------------------------------------------
# 생성자는 랜덤 노이즈 벡터를 입력받아 실제 이미지와 유사한 가짜 이미지를 생성합니다.
generator = Sequential()

# 입력층 & 은닉층
# input_dim=100: 100차원의 랜덤 노이즈 벡터를 입력받습니다.
# 128*7*7: 128개의 채널을 가진 7x7 크기의 이미지로 변환하기 위한 노드 수입니다.
# activation=LeakyReLU(0.2): ReLU의 변형으로, 음수 입력에 대해 0.2만큼의 기울기를 가집니다. (죽은 ReLU 방지)
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))

# 배치 정규화 (BatchNormalization)
# 학습 안정성을 높이고 학습 속도를 가속화합니다.
generator.add(BatchNormalization())

# 형태 변환 (Reshape)
# 1차원 배열을 (7, 7, 128) 형태의 3차원 텐서로 변환합니다. (가로, 세로, 채널)
generator.add(Reshape((7, 7, 128)))

# 업샘플링 (UpSampling2D)
# 이미지의 가로, 세로 크기를 2배로 늘립니다. (7x7 -> 14x14)
generator.add(UpSampling2D())

# 합성곱 층 (Conv2D)
# 커널 크기 5x5, 패딩 'same'으로 이미지 크기를 유지하며 특징을 추출합니다.
generator.add(Conv2D(64, kernel_size=5, padding='same'))

# 배치 정규화
generator.add(BatchNormalization())

# 활성화 함수 (LeakyReLU)
generator.add(Activation(LeakyReLU(0.2)))

# 업샘플링 (UpSampling2D)
# 이미지 크기를 다시 2배로 늘립니다. (14x14 -> 28x28)
generator.add(UpSampling2D())

# 출력층 (Conv2D)
# 28x28 크기의 이미지를 생성합니다. 채널 수는 1 (흑백)입니다.
# activation='tanh': 픽셀 값을 -1에서 1 사이로 정규화하여 출력합니다.
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))


# -----------------------------------------------------------------------------
# 2. 판별자 (Discriminator) 모델 설계
# -----------------------------------------------------------------------------
# 판별자는 입력된 이미지가 '실제'인지 '가짜(생성된 것)'인지 구별합니다.
discriminator = Sequential()

# 합성곱 층
# input_shape=(28,28,1): 28x28 크기의 흑백 이미지를 입력받습니다.
# strides=2: 필터를 2칸씩 이동시켜 이미지 크기를 줄입니다. (다운샘플링 효과)
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding="same"))

# 활성화 함수 (LeakyReLU)
discriminator.add(Activation(LeakyReLU(0.2)))

# 드롭아웃 (Dropout)
# 과적합을 방지하기 위해 30%의 뉴런을 무작위로 비활성화합니다.
discriminator.add(Dropout(0.3))

# 두 번째 합성곱 층
# strides=2를 사용하여 이미지 크기를 더욱 줄입니다.
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))

# 활성화 함수 (LeakyReLU)
discriminator.add(Activation(LeakyReLU(0.2)))

# 드롭아웃 (Dropout)
discriminator.add(Dropout(0.3))

# 평탄화 (Flatten)
# 3차원 특징 맵을 1차원 벡터로 변환하여 완전연결층에 전달합니다.
discriminator.add(Flatten())

# 출력층 (Dense)
# 1개의 노드로 출력하며, activation='sigmoid'를 사용하여 0(가짜) ~ 1(진짜) 사이의 확률을 출력합니다.
discriminator.add(Dense(1, activation='sigmoid'))

# 판별자 컴파일
# binary_crossentropy: 이진 분류(진짜/가짜) 문제에 적합한 손실 함수입니다.
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# GAN 모델 학습 시 판별자는 학습되지 않도록 고정합니다.
# (판별자는 fake/real 데이터를 따로 받아서 학습시킬 것이기 때문입니다.)
discriminator.trainable = False


# -----------------------------------------------------------------------------
# 3. GAN 모델 정의 (생성자 + 판별자 연결)
# -----------------------------------------------------------------------------
# 생성 모델을 학습시키기 위해 생성자와 판별자를 연결합니다.
# GAN 모델에서 판별자는 학습되지 않고(trainable=False), 생성자만 학습됩니다.

# 입력: 100차원 노이즈
ginput = Input(shape=(100,))

# 흐름: 입력 -> 생성자 -> 가짜 이미지 -> 판별자 -> 판별 결과 (0~1)
dis_output = discriminator(generator(ginput))

# 모델 생성 (Input: 노이즈, Output: 판별 결과)
gan = Model(ginput, dis_output)

# GAN 모델 컴파일
gan.compile(loss='binary_crossentropy', optimizer='adam')

# GAN 모델 요약 확인
gan.summary()


# =============================================================================
# [3] GAN 모델 학습 함수 (Training Function)
# =============================================================================
def gan_train(epoch, batch_size, saving_interval):
    # MNIST 데이터를 불러옵니다.
    # (X_train, _): 테스트 데이터는 사용하지 않고 학습 데이터(이미지)만 사용합니다.
    (X_train, _), (_, _) = mnist.load_data() 
    
    # 데이터 전처리
    # 차원 변환: (샘플수, 28, 28) -> (샘플수, 28, 28, 1)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    
    # 정규화: 픽셀 값(0~255)을 -1 ~ 1 사이로 변환합니다. (tanh 활성화 함수 범위에 맞춤)
    # (X - 127.5) / 127.5  =>  0은 -1로, 255는 1로 변환됨
    X_train = (X_train - 127.5) / 127.5 
    
    # 정답(True)과 가짜(Fake) 라벨 생성
    # 1: 진짜 이미지, 0: 가짜 이미지
    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for i in range(epoch):
        # ---------------------------------------------------------------------
        # [Step 1] 판별자(Discriminator) 학습
        # ---------------------------------------------------------------------
        # 실제 데이터에서 무작위로 배치를 추출합니다.
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        # 실제 이미지에 대해 판별자를 학습시킵니다. (정답: 1)
        d_loss_real = discriminator.train_on_batch(imgs, true)
        
        # 가상(생성된) 이미지를 생성합니다.
        # 정규분포(평균0, 표준편차1)를 따르는 100차원 노이즈 생성
        noise = np.random.normal(0, 1, (batch_size, 100))
        # 생성자를 통해 이미지 생성
        gen_imgs = generator.predict(noise, verbose=0)
        
        # 가짜 이미지에 대해 판별자를 학습시킵니다. (정답: 0)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        
        # 판별자의 전체 손실(Loss) 계산 (실제 손실과 가짜 손실의 평균)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------------------------------------------------------
        # [Step 2] 생성자(Generator) 학습 (GAN 모델을 통해)
        # ---------------------------------------------------------------------
        # 생성자는 판별자를 속이는 방향으로 학습해야 합니다.
        # 즉, 생성된 이미지가 판별자에 의해 '1(진짜)'로 분류되도록 학습합니다.
        g_loss = gan.train_on_batch(noise, true)
        
        # ---------------------------------------------------------------------
        # [Step 3] 로그 출력 및 이미지 저장
        # ---------------------------------------------------------------------
        if i % 100 == 0:
            print('epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)
            
        # saving_interval 마다 생성된 이미지를 저장합니다.
        if i % saving_interval == 0:
            # 테스트를 위한 노이즈 생성 (25개)
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)
            
            # 이미지 값을 다시 0 ~ 1 사이로 변환 (시각화를 위해)
            # -1 ~ 1  =>  0 ~ 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            # 5x5 그리드로 이미지 출력 및 저장
            fig, axs = plt.subplots(5, 5)
            count = 0
            for j in range(5):
                for k in range(5):
                    axs[j, k].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                    axs[j, k].axis('off')
                    count += 1
            # 이미지 파일로 저장
            fig.savefig("./gan_mnist_%d.png" % i)

# 함수 실행 (Epoch 2001회, 배치 크기 32, 저장 주기 200회)
gan_train(2001, 32, 200)



# =============================================================================
# [4] 오토인코더 (Autoencoder)
# =============================================================================
# 오토인코더는 입력 데이터를 압축(인코딩)했다가 다시 복원(디코딩)하는 신경망입니다.
# 입력과 출력이 동일해지도록 학습하며, 이 과정에서 데이터의 주요 특징을 학습합니다.

# 데이터 불러오기 및 전처리
(X_train, _), (X_test, _) = mnist.load_data()

# 0 ~ 1 사이 값으로 정규화
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255


# -----------------------------------------------------------------------------
# 1. 오토인코더 모델 설계
# -----------------------------------------------------------------------------
autoencoder = Sequential()

# [인코더 (Encoder)] : 입력 이미지의 차원을 줄여 특징을 추출하는 부분
# Conv2D: 16개 필터, 3x3 커널, ReLU 활성화
autoencoder.add(Conv2D(16, kernel_size=3, padding='same', input_shape=(28,28,1), activation='relu'))
# MaxPooling2D: 크기를 1/2로 축소 (28x28 -> 14x14)
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
# Conv2D: 8개 필터 (14x14)
autoencoder.add(Conv2D(8, kernel_size=3, activation='relu', padding='same'))
# MaxPooling2D: 크기를 1/2로 축소 (14x14 -> 7x7)
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
# Conv2D: 8개 필터, strides=2로 사용하여 크기를 더욱 축소 (7x7 -> 4x4)
# 이 부분이 잠재 공간(Latent Space)에 해당하는 압축된 특징입니다.
autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu'))

# [디코더 (Decoder)] : 압축된 특징을 다시 원래 이미지 크기로 복원하는 부분
# Conv2D: 특징 추출
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
# UpSampling2D: 크기를 2배로 확대 (4x4 -> 8x8)
autoencoder.add(UpSampling2D())
# Conv2D: 특징 추출
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
# UpSampling2D: 크기를 2배로 확대 (8x8 -> 16x16)
autoencoder.add(UpSampling2D())
# Conv2D: 특징 추출 (필터 수 16으로 증가)
autoencoder.add(Conv2D(16, kernel_size=3, activation='relu'))
# UpSampling2D: 크기를 확대 (14x14 -> 28x28) 
# 주의: 직전 레이어에서 padding='valid'(default)라면 크기가 14x14가 됨. 
# 여기서는 UpSampling으로 최종적으로 28x28로 맞춥니다.
autoencoder.add(UpSampling2D())
# 출력층 (Conv2D)
# 원래 이미지(28x28, 1채널)로 복원합니다.
# activation='sigmoid': 0~1 사이의 픽셀 값 출력
autoencoder.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

# 모델 구조 요약
autoencoder.summary()

# 모델 컴파일
# 입력과 출력의 차이(손실)를 줄이는 방향으로 학습
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 학습
# 입력(X_train)과 정답(X_train)이 동일합니다. (비지도 학습)
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_test, X_test))


# =============================================================================
# [5] 결과 시각화 (Visualization)
# =============================================================================
# 테스트 데이터셋에서 무작위로 5개의 이미지를 선택하여 복원 결과를 확인합니다.
random_test = np.random.randint(X_test.shape[0], size=5)

# 오토인코더를 통해 이미지 복원 (예측)
ae_imgs = autoencoder.predict(X_test)

# 그래프 그리기
plt.figure(figsize=(7, 2))

for i, image_idx in enumerate(random_test):
    # 원본 이미지 출력 (윗줄)
    ax = plt.subplot(2, 7, i + 1)
    plt.imshow(X_test[image_idx].reshape(28, 28)) # 흑백 이미지 출력
    ax.axis('off')
    
    # 복원된 이미지 출력 (아랫줄)
    ax = plt.subplot(2, 7, 7 + i +1)
    plt.imshow(ae_imgs[image_idx].reshape(28, 28)) # 복원된 이미지
    ax.axis('off')

# 전체 출력
plt.show()