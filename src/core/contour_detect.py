import cv2
import numpy as np

def get_lung_contour_mask(image):
    """
    获取胸腔的轮廓掩码
    
    :param image: 输入图像，可以是灰度图或三通道图像
    
    :return: 胸腔的轮廓掩码，1 代表胸腔内区域，-1 代表背景
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

