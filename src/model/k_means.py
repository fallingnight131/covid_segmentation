import numpy as np
import time
import logging
from scipy.spatial.distance import cdist  # 用于计算样本点与簇中心之间的欧氏距离
from scipy.sparse import coo_matrix  # 用于创建稀疏矩阵以优化簇中心的更新过程

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    def __init__(self, k, max_iter=500, tol=1e-4, random_state=42, verbose=True):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose  # 控制是否打印信息
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
        
        for i in range(self.max_iter):
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
            
            # 计算更新量
            center_shift = np.sum(np.abs(new_centers - self.centers))
            if center_shift < self.tol:
                break

            if self.verbose and i % 10 == 0:
                logging.info(f"已训练第 {i + 1} 轮，当前中心更新量: {center_shift:.4f}")
                
            self.centers = new_centers
            
        end_time = time.time()
        logging.info(f"K-means 训练总用时: {end_time - start_time:.2f} 秒")
    
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
    def __init__(self, k, batch_size=1000, max_iter=500, tol=1e-4, random_state=42, verbose=True):
        self.k = k
        self.batch_size = batch_size  # 每次只用 batch_size 个样本更新中心
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
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

        for i in range(self.max_iter):
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
            center_shift = np.sum(np.abs(new_centers - self.centers))
            if center_shift < self.tol:
                break

            if self.verbose and i % 10 == 0:
                logging.info(f"已训练第 {i + 1} 轮，当前中心更新量: {center_shift:.4f}")
                
            self.centers = new_centers
            
        end_time = time.time()  # 记录算法结束时间
        logging.info(f"K-means 训练总用时: {end_time - start_time:.2f} 秒")

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