import numpy as np
from model.k_means import CustomKMeans as KMeans # 导入自定义 K-Means
from util.data_process import preprocess
from core.contour_detect import get_lung_contour_mask



# 对图片的 KMeans 聚类 (仅使用灰度值)
def images_kmeans_train(images, k1=2):
    all_features = []  # 用于存储所有图像的像素特征

    # 1. 遍历所有图片，提取特征
    for img in images:
        img_processed = preprocess(img)
        if len(img_processed.shape) == 3:
            img_processed = img_processed.squeeze()

        in_lung_contour = get_lung_contour_mask(img)

        pixels = img_processed.flatten().reshape(-1, 1)  # 仅灰度值
        lung_contour_flat = in_lung_contour.flatten().reshape(-1, 1)  # 展平的肺部轮廓掩码
        
        # 组合特征 (灰度值 + 掩码)
        sub = pixels - lung_contour_flat
        all_features.append(sub)
        

    # 2. 合并所有图片的特征，并训练 KMeans
    all_features = np.vstack(all_features)  # 纵向堆叠，形成 (总像素数, 2) 的数组
    print(f"Total pixels: {all_features.shape[0]}")  # 打印总像素数量

    # 训练 KMeans（使用全局数据）
    k_means = KMeans(k=k1)
    k_means.fit(all_features)

    
    return k_means  # 返回训练好的 KMeans 模型
