import numpy as np
import matplotlib.pyplot as plt
import os
from util.data_process import  normalize_ct , align_labels, remove_noise_morphology, remove_noise_connected_components
from core.train import images_kmeans_train
from core.predict import images_kmeans_predict

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../data', 'images_radiopedia.npy')

# 检查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# 加载图像数据
num_images_to_process = 50 
start_num = 0
images = np.load(file_path)[start_num : start_num + num_images_to_process]  
images = normalize_ct(images)
images = images.squeeze()  
images_test = images[:50]

# 对图像进行 KMeans 聚类
kmeans_model = images_kmeans_train(images, k1=2, batch_size=512*512*100, max_iter=100, tol=1e-4, random_state=42)

# 预测图像的分割结果
labels = images_kmeans_predict(images_test, kmeans_model)

# 统一标签，防止颜色翻转
final_labels = align_labels(labels)

# 1. 形态学开运算去噪
labels_denoised = remove_noise_morphology(final_labels, operation="MORPH_CLOSE", kernel_size=3)
labels_denoised = remove_noise_morphology(labels_denoised, operation="MORPH_OPEN", kernel_size=3)

# 2. 连通区域分析去小噪声
labels_denoised = remove_noise_connected_components(labels_denoised, min_size=500)

# 查看最终结果
plt.figure(figsize=(15, 10))
for i, color_segmented in enumerate(labels_denoised):
    plt.subplot(5, 10, i + 1)
    plt.imshow(color_segmented, vmin=0, vmax=1)
    plt.title(f'Image {i+1} Segmentation')
    plt.axis('off')
plt.show()