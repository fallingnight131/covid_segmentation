import numpy as np

def compute_iou(test_masks, predict_masks):
    """
    计算 IoU（交并比）的平均值
    
    :param test_masks: 真实 mask (N, H, W)，可以是 list 或 NumPy 数组
    :param predict_masks: 预测 mask (N, H, W)，可以是 list 或 NumPy 数组
    
    :return: IoU 的平均值 (float)
    """
    # 转换为 NumPy 数组，确保正确形状
    test_masks = np.array(test_masks)
    predict_masks = np.array(predict_masks)

    if test_masks.shape != predict_masks.shape:
        raise ValueError(f"Shape mismatch: {test_masks.shape} vs {predict_masks.shape}")

    # 计算 IoU
    intersection = np.logical_and(test_masks, predict_masks).sum(axis=(1, 2))
    union = np.logical_or(test_masks, predict_masks).sum(axis=(1, 2))
    
    iou = intersection / (union + 1e-8)  # 避免除零

    return np.mean(iou)  # 返回 IoU 平均值