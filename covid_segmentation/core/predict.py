import numpy as np
from util.data_process import preprocess# 预测 KMeans 分割
from core.contour_detect import get_lung_contour_mask


def images_kmeans_predict(images, k_means):
    final_labels = []
    image_shapes = []
    all_features = []  # 存储所有图像的特征

    start_idx = 0
    
    # 1. 遍历所有图片，提取特征
    for img in images:
        img_processed = preprocess(img)
        if len(img_processed.shape) == 3:
            img_processed = img_processed.squeeze()

        in_lung_contour = get_lung_contour_mask(img)

        pixels = img_processed.flatten().reshape(-1, 1)  # 仅灰度值
        lung_contour_flat = in_lung_contour.flatten().reshape(-1, 1)  # 肺部掩码
        
        # 组合特征 (灰度值 + 掩码)
        sub = pixels - lung_contour_flat
        all_features.append(sub)

        image_shapes.append(img_processed.shape[:2])  # 记录原始图像尺寸

    # 2. 预测 KMeans
    all_features = np.vstack(all_features)
    labels_all = k_means.predict(all_features)

    # 3. 逐张图片还原标签
    for shape in image_shapes:
        h, w = shape
        num_pixels = h * w

        # 取出当前图片的像素标签
        labels = labels_all[start_idx: start_idx + num_pixels]

        # 还原回图片形状
        img_labels = labels.reshape((h, w))
        final_labels.append(img_labels)

        start_idx += num_pixels  # 更新索引

    return final_labels  # 返回分割结果