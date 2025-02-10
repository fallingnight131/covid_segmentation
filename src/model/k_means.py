import numpy as np
import time
from scipy.spatial.distance import cdist  # 用于计算样本点与簇中心之间的欧氏距离
from scipy.sparse import coo_matrix  # 用于创建稀疏矩阵以优化簇中心的更新过程

class CustomKMeans:
    """
    自定义 KMeans 类
    
    参数:
    k: int, 簇的数量
    max_iter: int, 最大迭代次数
    tol: float, 中心更新的阈值
    random_state: int, 随机种子
    
    属性:
    centers: ndarray, shape (k, n_features), 簇中心
    labels: ndarray, shape (n_samples,), 样本的标签
    """
    def __init__(self, k, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.labels = None
    
    def fit(self, X):
        """
        训练 KMeans 模型
        
        :param X: ndarray, shape (n_samples, n_features), 输入数据
        
        :return: None
        """
        start_time = time.time()
        np.random.seed(self.random_state)
        
        self.centers = self._initialize_centers(X)
        
        for _ in range(self.max_iter):
            X_squared = np.sum(X**2, axis=1).reshape(-1, 1)
            centers_squared = np.sum(self.centers**2, axis=1).reshape(1, -1)
            distances = X_squared + centers_squared - 2 * np.dot(X, self.centers.T)
            self.labels = np.argmin(distances, axis=1)
            
            rows, cols = np.arange(X.shape[0]), self.labels
            data = np.ones(X.shape[0])
            sparse_matrix = coo_matrix((data, (rows, cols)), shape=(X.shape[0], self.k))
            cluster_sums = sparse_matrix.T @ X
            cluster_counts = sparse_matrix.sum(axis=0).A1
            new_centers = cluster_sums / cluster_counts[:, np.newaxis]
            
            if np.all(np.abs(new_centers - self.centers) < self.tol):
                break
            
            self.centers = new_centers
        
        end_time = time.time()
        print(f"Optimized K-means execution time: {end_time - start_time:.2f} seconds")
    
    def predict(self, X):
        """
        预测输入数据的标签
        
        :param X: ndarray, shape (n_samples, n_features), 输入数据
        
        :return: labels: ndarray, shape (n_samples,), 预测的标签
        """
        X_squared = np.sum(X**2, axis=1).reshape(-1, 1)
        centers_squared = np.sum(self.centers**2, axis=1).reshape(1, -1)
        distances = X_squared + centers_squared - 2 * np.dot(X, self.centers.T)
        return np.argmin(distances, axis=1)
    
    def _initialize_centers(self, X):
        """
        使用 KMeans++ 初始化中心
        
        :param X: ndarray, shape (n_samples, n_features), 输入数据
        
        :return: centers: ndarray, shape (k, n_features), 初始化的簇中心
        """
        n_samples, _ = X.shape
        centers = [X[np.random.randint(0, n_samples)]]
        for _ in range(1, self.k):
            distances = np.min(cdist(X, np.array(centers)), axis=1)
            probs = distances / np.sum(distances)
            new_center = X[np.random.choice(n_samples, p=probs)]
            centers.append(new_center)
        return np.array(centers)

class MiniBatchKMeans:
    """
    自定义 MiniBatchKMeans 类
    
    参数:
    k: int, 簇的数量
    batch_size: int, 每次更新中心的样本数量
    max_iter: int, 最大迭代次数
    tol: float, 中心更新的阈值
    random_state: int, 随机种子
    
    属性:
    centers: ndarray, shape (k, n_features), 簇中心
    labels: ndarray, shape (n_samples,), 样本的标签
    """
    def __init__(self, k, batch_size=1000, max_iter=1000, tol=1e-4, random_state=42):
        self.k = k
        self.batch_size = batch_size  # 每次只用 batch_size 个样本更新中心
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.labels = None
    
    def fit(self, X):
        """
        训练 MiniBatchKMeans 模型
        
        :param X: ndarray, shape (n_samples, n_features), 输入数据
        
        :return: None
        """
        start_time = time.time()  # 记录算法开始时间
        np.random.seed(self.random_state)
        self.centers = self._initialize_centers(X)
        n_samples = X.shape[0]

        for _ in range(self.max_iter):
            # **随机采样 batch_size 个数据点**
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[batch_indices]

            # 计算 batch 数据到中心的欧氏距离
            X_squared = np.sum(X_batch**2, axis=1).reshape(-1, 1)
            centers_squared = np.sum(self.centers**2, axis=1).reshape(1, -1)
            distances = X_squared + centers_squared - 2 * np.dot(X_batch, self.centers.T)
            batch_labels = np.argmin(distances, axis=1)

            # 计算小批量的簇中心更新
            rows, cols = np.arange(X_batch.shape[0]), batch_labels
            data = np.ones(X_batch.shape[0])
            sparse_matrix = coo_matrix((data, (rows, cols)), shape=(X_batch.shape[0], self.k))
            cluster_sums = sparse_matrix.T @ X_batch
            cluster_counts = sparse_matrix.sum(axis=0).A1 + 1e-8  # 防止除零错误
            new_centers = cluster_sums / cluster_counts[:, np.newaxis]

            # 计算更新量
            if np.all(np.abs(new_centers - self.centers) < self.tol):
                break

            self.centers = new_centers
            
        end_time = time.time()  # 记录算法结束时间
        print(f"Optimized K-means execution time: {end_time - start_time:.2f} seconds")  # 打印算法执行时间

    def predict(self, X):
        """
        预测输入数据的标签
        
        :param X: ndarray, shape (n_samples, n_features), 输入数据
        
        :return: labels: ndarray, shape (n_samples,), 预测的标签
        """
        X_squared = np.sum(X**2, axis=1).reshape(-1, 1)
        centers_squared = np.sum(self.centers**2, axis=1).reshape(1, -1)
        distances = X_squared + centers_squared - 2 * np.dot(X, self.centers.T)
        return np.argmin(distances, axis=1)

    def _initialize_centers(self, X):
        """
        使用 KMeans++ 初始化中心
        
        :param X: ndarray, shape (n_samples, n_features), 输入数据
        
        :return: centers: ndarray, shape (k, n_features), 初始化的簇中心
        """
        n_samples, _ = X.shape
        centers = [X[np.random.randint(0, n_samples)]]
        for _ in range(1, self.k):
            distances = np.min(cdist(X, np.array(centers)), axis=1)
            probs = distances / np.sum(distances)
            new_center = X[np.random.choice(n_samples, p=probs)]
            centers.append(new_center)
        return np.array(centers)