import numpy as np
import cv2

def preprocess(img):
    """
    图像预处理：归一化和标准化。
    
    :param img: 输入图像
    
    :return: 预处理后的图像
    """
    img = img / 255.0  # 归一化到 [0, 1]
    return (img - img.mean()) / (img.std() + 1e-8)  # 标准化

def normalize_ct(image, window_width=1500, window_level=-600):
    """
    归一化 CT 图像到 0-255。
    
    :param image: 输入图像
    :param window_width: 窗口宽度
    :param window_level: 窗口层次
    
    :return: 归一化后的图像
    """
    min_hu = window_level - (window_width / 2)
    max_hu = window_level + (window_width / 2)
    
    image = np.clip(image, min_hu, max_hu)  # 限制在窗口范围内
    image = (image - min_hu) / (max_hu - min_hu) * 255  # 归一化到 0-255
    return image.astype(np.uint8)  # 转换为 uint8 以适应 OpenCV

def invert_binary_images(images):
    """
    对所有输入的二值图像进行黑白翻转（0 变 255，255 变 0）。
    
    :param images: 输入图像列表
    
    :return: 翻转后的图像列表
    """
    return [255 - img for img in images]

def align_labels(final_labels):
    """
    统一标签，防止颜色翻转。
    
    :param final_labels: 输入的标签列表
    
    :return: 统一后的标签列表
    """
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

def convert_mask_auto(mask):
    """
    在 (N, H, W) 二值掩码 和 (N, H, W, 4) 四通道掩码 之间转换。

    规则：
    - (N, H, W) → (N, H, W, 4)：
        - 若单通道值为 1，则四通道的最后一个通道为 True，其余通道为 False
        - 若单通道值为 0，则四通道所有通道为 False
    - (N, H, W, 4) → (N, H, W)：
        - 若四通道的最后一个通道为 True，则单通道值为 1
        - 否则单通道值为 0

    :param mask: NumPy 数组，形状 (N, H, W) 或 (N, H, W, 4)
    :return: 转换后的 NumPy 数组
    """
    mask = np.array(mask)  # 确保是 NumPy 数组
    shape = mask.shape

    # **单通道 (N, H, W) → 四通道 (N, H, W, 4)**
    if len(shape) == 3:
        N, H, W = shape
        new_mask = np.zeros((N, H, W, 4), dtype=bool)  # 全部初始化为 False
        
        # 单通道为 1 的地方，四通道最后一维设为 True
        new_mask[..., 3] = (mask == 1)
        
        return new_mask

    # **四通道 (N, H, W, 4) → 单通道 (N, H, W)**
    elif len(shape) == 4 and shape[-1] == 4:
        # 取最后一个通道作为单通道掩码
        binary_mask = mask[..., 3].astype(np.uint8)  # (N, H, W)
        return binary_mask

    else:
        raise ValueError("输入 mask 形状必须是 (N, H, W) 或 (N, H, W, 4)！")

    
def remove_noise_morphology(labels, kernel_size=5, operation="MORPH_OPEN"):
    """
    使用形态学操作去除标签中的噪声。
    
    :param labels: 输入的标签列表
    :param kernel_size: 形态学核大小
    :param operation: 形态学操作，"MORPH_OPEN" 或 "MORPH_CLOSE"
    
    :return: 去噪后的标签列表
    """
    denoised_labels = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    if operation == "MORPH_OPEN":
        denoised_labels = [cv2.morphologyEx(label.astype(np.uint8), cv2.MORPH_OPEN, kernel) for label in labels]
    elif operation == "MORPH_CLOSE":
        denoised_labels = [cv2.morphologyEx(label.astype(np.uint8), cv2.MORPH_CLOSE, kernel) for label in labels]
    else:
        raise ValueError(f"Unsupported morphological operation: {operation}")
    return denoised_labels

def remove_noise_connected_components(labels, min_size=500):
    """
    使用连通区域分析去除标签中的噪声。
    
    :param labels: 输入的标签列表
    :param min_size: 最小区域面积
    
    :return: 去噪后的标签列表
    """
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


