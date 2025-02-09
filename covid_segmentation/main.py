import numpy as np
import matplotlib.pyplot as plt
import os
from util.data_process import  normalize_ct , align_labels, remove_noise_morphology, remove_noise_connected_components
from core.train import images_kmeans_train
from core.predict import images_kmeans_predict

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'data', 'images_radiopedia.npy')

# 检查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# 加载图像数据
num_images_to_process = 100 
start_num = 20
images = np.load(file_path)[start_num : start_num + num_images_to_process]  
images = normalize_ct(images)
images = images.squeeze()  

# 对图像进行 KMeans 聚类
kmeans_model = images_kmeans_train(images, k1=2)

# 预测图像的分割结果
labels = images_kmeans_predict(images, kmeans_model)

# 统一标签，防止颜色翻转
final_labels = align_labels(labels)

# 1. 形态学开运算去噪
labels_denoised = remove_noise_morphology(final_labels, operation="MORPH_OPEN")
labels_denoised = remove_noise_morphology(labels_denoised, operation="MORPH_CLOSE")

# 2. 连通区域分析去小噪声
labels_denoised = remove_noise_connected_components(labels_denoised, min_size=500)

# 选择几张图像查看最终结果
selected_indices = [15, 20, 25, 32, 35]
plt.figure(figsize=(15, 5))
for i, idx in enumerate(selected_indices):
    plt.subplot(1, len(selected_indices), i + 1)
    plt.imshow(labels_denoised[idx], vmin=0, vmax=1)
    plt.title(f'Image {idx+1} Segmentation')
    plt.axis('off')
plt.show()
