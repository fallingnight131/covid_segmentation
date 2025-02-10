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

# 自动转换 mask 类型
def convert_mask_auto(mask):

    shape = mask.shape
    
    # 判断输入是否为 (N, H, W)
    if len(shape) == 3:
        N, H, W = shape
        new_mask = np.zeros((N, H, W, 4), dtype=np.uint8)

        # 假设 mask 只有前景 (1) 和背景 (0)，填充前三个通道为前景
        new_mask[..., 0] = (mask == 1)  # 第一个通道：类别 1
        new_mask[..., 1] = (mask == 1)  # 第二个通道：类别 2
        new_mask[..., 2] = (mask == 1)  # 第三个通道：类别 3
        new_mask[..., 3] = (mask == 0)  # 第四个通道：背景 (1 代表背景)

        # 将背景设为 1，前景设为 0
        new_mask = 1 - new_mask

        return new_mask
    
    # 判断输入是否为 (N, H, W, 4)
    elif len(shape) == 4 and shape[-1] == 4:
        # 取前三个通道的最大值，判断是否有前景
        foreground = np.max(mask[..., :3], axis=-1)  # (N, H, W)

        # 背景通道
        background = mask[..., 3]  # (N, H, W)

        # 生成二值 mask：前景为 0，背景为 1
        binary_mask = np.where(background, 1, 0).astype(np.uint8)

        return binary_mask

    else:
        raise ValueError("输入 mask 形状必须是 (N, H, W) 或 (N, H, W, 4)！")
    
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


