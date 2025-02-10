import numpy as np
import matplotlib.pyplot as plt
import os
from util.data_process import  normalize_ct , align_labels, remove_noise_morphology, remove_noise_connected_components
from core.train import images_kmeans_train
from core.predict import images_kmeans_predict

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
train_image_path = os.path.join(current_dir, '../data/split', 'train_images.npy')
test_image_path = os.path.join(current_dir, '../data/split', 'test_images.npy')
test_mask_path = os.path.join(current_dir, '../data/split', 'test_masks.npy')
model_path = os.path.join(current_dir, '../model/', 'kmeans_model.pkl')

# 加载图像数据
num_images_to_process = 50 
start_num = 0
images_train = np.load(train_image_path)[start_num : start_num + num_images_to_process]  
images_train = normalize_ct(images_train)
images_train = images_train.squeeze()  

images_test = np.load(test_image_path)
images_test = normalize_ct(images_test)
images_test = images_test.squeeze()

# 对图像进行 KMeans 聚类
images_kmeans_train(images_train, k1=2, batch_size=512*512*100, 
                                   max_iter=100, tol=1e-4, random_state=42, 
                                   model_path=model_path)

# 预测图像的分割结果
labels = images_kmeans_predict(images_test, model_path=model_path)

# 统一标签，防止颜色翻转
final_labels = align_labels(labels)

# 1. 形态学开运算去噪
labels_denoised = remove_noise_morphology(final_labels, operation="MORPH_CLOSE", kernel_size=3)
labels_denoised = remove_noise_morphology(labels_denoised, operation="MORPH_OPEN", kernel_size=3)

# 2. 连通区域分析去小噪声
labels_denoised = remove_noise_connected_components(labels_denoised, min_size=500)

# 查看最终结果
num_images = len(labels_denoised)
half_num_images = num_images // 2  # 分成两部分

# 第一部分
plt.figure(figsize=(15, 15))
for i, color_segmented in enumerate(labels_denoised[:half_num_images]):
    grid_size = int(np.ceil(np.sqrt(half_num_images)))  # 动态计算网格大小
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(color_segmented, vmin=0, vmax=1)
    plt.title(f'Image {i+1} Segmentation')
    plt.axis('off')
plt.show()

# 第二部分
plt.figure(figsize=(15, 15))
for i, color_segmented in enumerate(labels_denoised[half_num_images:]):
    grid_size = int(np.ceil(np.sqrt(num_images - half_num_images)))  # 动态计算网格大小
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(color_segmented, vmin=0, vmax=1)
    plt.title(f'Image {half_num_images + i + 1} Segmentation')
    plt.axis('off')
plt.show()