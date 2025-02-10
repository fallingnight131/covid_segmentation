import numpy as np
import matplotlib.pyplot as plt

def display_images_paginated(images_test, test_masks, 
                             labels_denoised, images_per_page=10):
    """
    分页显示原始图像、真实掩码和预测掩码。
    
    :param images_test: 测试图像列表
    :param test_masks: 真实掩码列表
    :param labels_denoised: 预测掩码列表
    :param images_per_page: 每页显示的图像数量（默认为10）
    """
    num_images = len(labels_denoised)
    num_pages = int(np.ceil(num_images / images_per_page))
    
    for page in range(num_pages):
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, num_images)
        
        # 动态调整画布大小
        fig_height = 9 * (end_idx - start_idx) / images_per_page
        plt.figure(figsize=(15, fig_height))
        
        for i, idx in enumerate(range(start_idx, end_idx)):
            plt.subplot(end_idx - start_idx, 3, i * 3 + 1)
            plt.imshow(images_test[idx], cmap='gray')
            plt.title(f'Original Image {idx+1}')
            plt.axis('off')

            plt.subplot(end_idx - start_idx, 3, i * 3 + 2)
            plt.imshow(test_masks[idx], vmin=0, vmax=1)
            plt.title(f'True Mask {idx+1}')
            plt.axis('off')

            plt.subplot(end_idx - start_idx, 3, i * 3 + 3)
            plt.imshow(labels_denoised[idx], vmin=0, vmax=1)
            plt.title(f'Predicted Mask {idx+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
