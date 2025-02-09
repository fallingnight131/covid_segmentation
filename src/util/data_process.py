import numpy as np
import cv2

# 预处理：归一化图像
def preprocess(img):
    img = img / 255.0  # 归一化到 [0, 1]
    return (img - img.mean()) / (img.std() + 1e-8)  # 标准化

# 归一化 CT 图像到 0-255 
def normalize_ct(image, window_width=1500, window_level=-600):
    min_hu = window_level - (window_width / 2)
    max_hu = window_level + (window_width / 2)
    
    image = np.clip(image, min_hu, max_hu)  # 限制在窗口范围内
    image = (image - min_hu) / (max_hu - min_hu) * 255  # 归一化到 0-255
    return image.astype(np.uint8)  # 转换为 uint8 以适应 OpenCV

# 对所有输入的二值图像进行黑白翻转（0变255，255变0）。
def invert_binary_images(images):
    return [255 - img for img in images]

# 统一标签，防止颜色翻转
def align_labels(final_labels):
    
    # 计算第一张图像的背景比例（0 的比例）
    first_label = final_labels[0]
    bg_ratio = np.mean(first_label == 1)  # 计算背景的像素比例

    # 若背景比例低于 50%，说明背景和前景反了，先翻转第一张图像
    if bg_ratio < 0.5:
        final_labels[0] = 1 - final_labels[0]  

    # 以第一张图片（可能已反转）为基准
    ref_label = final_labels[0]

    # 依次调整剩余图片
    for i in range(1, len(final_labels)):
        if np.mean(final_labels[i] == ref_label) < 0.5:  # 若相似度 < 0.5，则翻转
            final_labels[i] = 1 - final_labels[i]

    return final_labels

# 形态学去噪
def remove_noise_morphology(labels, kernel_size=5, operation="MORPH_OPEN"):
    denoised_labels = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    if operation == "MORPH_OPEN":
        denoised_labels = [cv2.morphologyEx(label.astype(np.uint8), cv2.MORPH_OPEN, kernel) for label in labels]
    elif operation == "MORPH_CLOSE":
        denoised_labels = [cv2.morphologyEx(label.astype(np.uint8), cv2.MORPH_CLOSE, kernel) for label in labels]
    else:
        raise ValueError(f"Unsupported morphological operation: {operation}")
    return denoised_labels

# 连通区域分析去小噪声
def remove_noise_connected_components(labels, min_size=500):
    denoised_labels = []
    for label in labels:
        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(label.astype(np.uint8))

        # 创建新掩码，只保留大于 min_size 的区域
        mask = np.zeros_like(label, dtype=np.uint8)
        for i in range(1, num_labels):  # 0 是背景，不处理
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[components == i] = 1  # 仅保留大区域

        denoised_labels.append(mask)
    return denoised_labels


