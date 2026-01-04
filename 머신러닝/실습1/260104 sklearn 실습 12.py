# =============================================================================
# [1] 라이브러리 임포트 (Library Import)
# =============================================================================
# 머신러닝 비지도 학습(차원 축소 및 클러스터링) 실습을 위한 라이브러리들을 모듈별로 불러옵니다.

# 1. 시각화 및 데이터 핸들링 라이브러리
import requests                  # 인터넷에서 이미지 등 파일 다운로드
from io import BytesIO           # 다운로드 받은 바이너리 데이터를 메모리 상에서 파일처럼 처리
from PIL import Image            # 이미지 파일 열기 및 처리
import pandas as pd              # 데이터프레임(표) 조작
import numpy as np               # 수치 연산 및 배열 처리
import matplotlib.pyplot as plt  # 그래프 그리기
from matplotlib import cm        # 컬러맵 처리 (Silhouette 분석 시 사용)
import seaborn as sns            # 예쁜 통계 차트 그리기

# 2. 사이킷런(Scikit-learn) - 데이터셋 및 기본 도구
from sklearn import datasets             # 내장 데이터셋 (iris 등) 로드
from sklearn.decomposition import PCA    # 주성분 분석 (Principal Component Analysis)
from sklearn.preprocessing import StandardScaler # 데이터 표준화 (스케일링)

# 3. 3차원 시각화 도구
from mpl_toolkits.mplot3d import Axes3D  # 3D 산점도 그리기

# 4. 차원 축소 알고리즘 (Dimensionality Reduction)
# LDA: 지도 학습 기반 차원 축소 (클래스 간 분산 최대화)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# TruncatedSVD: 특이값 분해 (희소 행렬에 유리)
from sklearn.decomposition import TruncatedSVD

# 5. 클러스터링 알고리즘 (Clustering)
from sklearn.cluster import KMeans       # K-평균 군집화
from sklearn.cluster import DBSCAN       # 밀도 기반 군집화 (DBSCAN)

# 6. 클러스터링 평가 지표
from sklearn.metrics import silhouette_samples, silhouette_score # 실루엣 계수 (군집화 품질 평가)


# =============================================================================
# [2] 데이터 로드 및 확인 (Data Loading & Inspection)
# =============================================================================

# 붓꽃(Iris) 데이터셋 로드
iris = datasets.load_iris()
# 데이터(특징) 부분만 추출
data = iris['data']
# 상위 5개 데이터 출력
print(data[:5])

# 데이터를 DataFrame으로 변환 (컬럼 이름 지정)
df = pd.DataFrame(data, columns=iris['feature_names'])
print(df.head())

# 타겟(정답, 꽃의 종류) 정보 추가
df['target'] = iris['target']
print(df.head())


# =============================================================================
# [3] 차원 축소: PCA (Principal Component Analysis)
# =============================================================================
# 고차원 데이터를 정보 손실을 최소화하면서 2차원 또는 3차원으로 줄입니다.

# n_components=2: 2개의 주성분(2차원)으로 축소하겠다는 설정
pca = PCA(n_components=2)

# 데이터 표준화: PCA는 스케일에 민감하므로 StandardScaler로 평균 0, 분산 1로 변환
# 타겟 컬럼은 제외하고 특징 컬럼들만 사용
data_scaled = StandardScaler().fit_transform(df.loc[:, 'sepal length (cm)': 'petal width (cm)'])

# 표준화된 데이터를 PCA로 변환 (4차원 -> 2차원)
pca_data = pca.fit_transform(data_scaled)

# 변환 전후 데이터 비교 출력
print(data_scaled[:5])
print(pca_data[:5])


# PCA 결과 2차원 산점도 시각화
# x축: 첫 번째 주성분, y축: 두 번째 주성분, 색상: 꽃의 종류(target)
print(plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['target']))

# n_components=0.99: 전체 분산의 99%를 설명할 수 있는 개수만큼 차원 축소
pca = PCA(n_components=0.99)
pca_data = pca.fit_transform(data_scaled)
print(pca_data[:5]) # 4개 차원이 다 필요할 수도, 줄어들 수도 있음


# =============================================================================
# [4] 3차원 시각화 (3D Visualization)
# =============================================================================
# PCA 결과를 3차원 공간에 시각화해 봅니다. (n_components=3인 경우 가정)
# 여기서는 pca_data가 위에서 다시 계산되어 차원이 달라졌을 수 있으나, 3열(index 2)이 있다면 3차원 가능

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # 3D 축 생성
sample_size = 50
# x, y, z 좌표에 각각 주성분 1, 2, 3 사용 (데이터가 3차원 이상이어야 함)
# 만약 pca_data 열이 부족하면 에러 발생 가능 (위에서 0.99로 해서 복원력에 따라 다름)
try:
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], alpha=0.6, c=df['target'])
except:
    print("3차원 데이터가 부족하여 시각화 건너뜀")

plt.savefig('./tmp.svg') # 결과 그림 저장
plt.title("ax.plot")
plt.show()


# =============================================================================
# [5] 차원 축소: LDA (Linear Discriminant Analysis)
# =============================================================================
# 지도 학습 방식의 차원 축소로, 클래스 간의 분리를 최대로 하는 축을 찾습니다.

print(df.head())

# n_components=2: 2차원으로 축소
lda = LinearDiscriminantAnalysis(n_components=2)
# 데이터 스케일링 (PCA 때와 동일하게 다시 수행)
data_scaled = StandardScaler().fit_transform(df.loc[:, 'sepal length (cm)': 'petal width (cm)'])

# LDA는 정답(target)이 필요함! (PCA와의 가장 큰 차이점)
lda_data = lda.fit_transform(data_scaled, df['target'])
print(lda_data[:5])

# 비교를 위한 PCA 시각화 (위에서 이미 했지만 다시 그림)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['target'])
plt.title("PCA Result")
plt.show()

# LDA 시각화: 클래스들이 더 잘 뭉치고, 서로 잘 떨어져 있는지 확인
plt.scatter(lda_data[:, 0], lda_data[:, 1], c=df['target'])
plt.title("LDA Result")
plt.show()


# =============================================================================
# [6] 차원 축소: SVD (Singular Value Decomposition)
# =============================================================================
# 특이값 분해를 이용한 차원 축소. PCA와 유사하지만 데이터 중심화 과정 등이 다릅니다.

print(df.head())
# 스케일링
data_scaled = StandardScaler().fit_transform(df.loc[:, 'sepal length (cm)': 'petal width (cm)'])

# TruncatedSVD 객체 생성 (2차원)
svd = TruncatedSVD(n_components=2)
svd_data = svd.fit_transform(data_scaled)

# 3가지 차원 축소 방법 비교 시각화
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['target'])
plt.title("PCA")
plt.show()

plt.scatter(lda_data[:, 0], lda_data[:, 1], c=df['target'])
plt.title("LDA")
plt.show()

plt.scatter(svd_data[:, 0], svd_data[:, 1], c=df['target'])
plt.title("SVD")
plt.show()


# =============================================================================
# [7] 군집화: K-Means Clustering
# =============================================================================
# 데이터를 K개의 그룹으로 묶는 가장 대표적인 비지도 학습 알고리즘입니다.

# n_clusters=3: 3개의 그룹으로 묶겠다 (붓꽃 종류가 3개니까)
kmeans = KMeans(n_clusters=3)
# 군집화 수행 (fit_transform은 거리 반환, fit_predict는 라벨 반환)
cluster_data = kmeans.fit_transform(df.loc[:, 'sepal length (cm)': 'petal width (cm)']) # 중심까지의 거리 반환

print(cluster_data[:5]) # 각 클러스터 중심까지의 거리
print(kmeans.labels_)   # 예측한 클러스터 번호 (0, 1, 2)


# K-Means가 예측한 클러스터별 데이터 개수 시각화
sns.countplot(x = kmeans.labels_)
plt.title("K-Means Prediction Clusters")
plt.show()

# 실제 정답(Target)의 데이터 개수 시각화 (비교용)
sns.countplot(x = df['target'])
plt.title("Actual Target Class")
plt.show()

print(kmeans)

# 하이퍼파라미터 변경 실험 (max_iter=500: 반복 횟수 늘림)
kmeans = KMeans(n_clusters=3, max_iter=500)
cluster_data = kmeans.fit_transform(df.loc[:, 'sepal length (cm)': 'petal width (cm)'])
sns.countplot(x = kmeans.labels_)
plt.show()


# 아웃라이어(이상치) 관련 이미지 자료 로드
try:
    response = requests.get('https://image.slidesharecdn.com/pydatanyc2015-151119175854-lva1-app6891/95/pydata-nyc-2015-automatically-detecting-outliers-with-datadog-26-638.jpg')
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
except:
    print("이미지 로드 실패")


# =============================================================================
# [8] 군집화: DBSCAN (Density-Based Spatial Clustering)
# =============================================================================
# 밀도 기반 군집화로, 기하학적인 모양의 군집도 잘 찾아내며 이상치(노이즈) 처리에 강합니다.

# eps=0.3: 이웃으로 인정하는 거리 반경
# min_samples=2: 군집이 되기 위한 최소 데이터 개수
dbscan = DBSCAN(eps=0.3, min_samples=2)

# 군집화 수행 및 예측
dbscan_data = dbscan.fit_predict(df.loc[:, 'sepal length (cm)': 'petal width (cm)'])

# 결과 확인 (-1은 노이즈/이상치로 분류된 데이터)
print(dbscan_data)


# =============================================================================
# [9] 군집화 평가: 실루엣 분석 (Silhouette Analysis)
# =============================================================================
# 군집화가 얼마나 잘 되었는지(군집 내 거리는 가깝고, 군집 간 거리는 먼지) 평가합니다.

# 전체 데이터의 평균 실루엣 계수 계산 output: -1 ~ 1 (1에 가까울수록 좋음)
score = silhouette_score(data_scaled, kmeans.labels_)
print(score)

# 개별 데이터의 실루엣 계수 계산
samples = silhouette_samples(data_scaled, kmeans.labels_)
print(samples[:5])


# 실루엣 분석 시각화 함수 (scikit-learn 문서 예제 활용)
def plot_silhouette(X, num_cluesters):
    for n_clusters in num_cluesters:
        # 1행 2열의 서브플롯 생성 (왼쪽: 실루엣 플롯, 오른쪽: 클러스터 산점도)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        
        # 1번 플롯: 실루엣 플롯 설정
        # 실루엣 계수 범위는 -1 ~ 1 이지만, 보통 0 이상인 경우가 많아 -0.1부터 표시
        ax1.set_xlim([-0.1, 1])
        # y축 범위 설정 (군집 사이 띄우기 위함)
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        
        # KMeans 군집화 수행
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        
        # 평균 실루엣 점수 계산 및 출력
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
        "The average silhouette_score is :", silhouette_avg)
        
        # 개별 샘플의 실루엣 점수 계산
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        y_lower = 10
        for i in range(n_clusters):
            # i번째 군집에 속하는 샘플들의 실루엣 점수만 가져와서 정렬
            ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # 색상 설정
            color = cm.nipy_spectral(float(i) / n_clusters)
            # 실루엣 그래프 그리기 (채우기)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
            0, ith_cluster_silhouette_values,
            facecolor=color, edgecolor=color, alpha=0.7)
            
            # 군집 번호 텍스트 표시
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # 다음 군집 그래프를 위해 y축 시작 위치 이동
            y_lower = y_upper + 10 # 10만큼 띄움
            
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        
        # 평균 실루엣 점수를 빨간 점선으로 표시
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        ax1.set_yticks([]) # y축 눈금 제거
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        
        # 2번 플롯: 클러스터 산점도 시각화
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
        
        # 클러스터 중심점(Centroids) 표시
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
            
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
        "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
        plt.show()


# 클러스터 개수(2, 3, 4, 5)를 바꿔가며 실루엣 분석 수행 및 시각화
plot_silhouette(data_scaled, [2, 3, 4, 5])
