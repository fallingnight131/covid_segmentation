import numpy as np
import cv2
from model.k_means import CustomKMeans as KMeans # 导入自定义 K-Means
from util.data_process import preprocess

def get_lung_contour_mask(image):
    """
    输入：肺部 CT / X-ray 图像，输出整个胸腔内的二值掩码
    :param image: 输入的肺部图像 (numpy 数组，单通道或三通道)
    :return: 二值掩码 (numpy 数组)，1 表示胸腔内部区域，-1 表示背景
    """
    # 1. 预处理：转换为灰度图
    if len(image.shape) == 3:  # 如果是三通道图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 2. 二值化处理（Otsu 阈值）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 轮廓提取
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.full_like(binary, -1, dtype=np.int8)  # 没找到轮廓，返回全 -1 矩阵

    # 4. 选择面积最大的轮廓（即胸腔的轮廓）
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. 生成二值掩码（填充整个胸腔）
    mask = np.full_like(binary, -1, dtype=np.int8)  # 创建全 -1 掩码
    cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)  # 填充胸腔内部

    return mask  # 1 代表胸腔内区域，-1 代表背景



# 对图片的 KMeans 聚类 (仅使用灰度值)
def images_kmeans(images, k1=2):
    final_labels = []
    image_shapes = []
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
        
        # 记录原始图像尺寸
        image_shapes.append(img_processed.shape[:2])  

    # 2. 合并所有图片的特征，并训练 KMeans
    all_features = np.vstack(all_features)  # 纵向堆叠，形成 (总像素数, 2) 的数组
    print(f"Total pixels: {all_features.shape[0]}")  # 打印总像素数量

    # 训练 KMeans（使用全局数据）
    k_means = KMeans(k=k1)
    k_means.fit(all_features)

    # 3. 逐张图片进行预测
    start_idx = 0
    for shape in image_shapes:
        h, w = shape
        num_pixels = h * w

        # 取出当前图片的特征
        img_features = all_features[start_idx : start_idx + num_pixels]

        # 预测当前图片的像素类别
        labels = k_means.predict(img_features)

        # 还原回图片形状
        img_labels = labels.reshape((h, w))
        final_labels.append(img_labels)

        start_idx += num_pixels  # 更新索引

    return final_labels, image_shapes
