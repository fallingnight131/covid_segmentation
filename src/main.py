import numpy as np
import matplotlib.pyplot as plt
import os
from util.data_process import  normalize_ct , align_labels, remove_noise_morphology, remove_noise_connected_components, convert_mask_auto
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

# 加载测试集的 mask
test_masks = np.load(test_mask_path)
test_masks = convert_mask_auto(test_masks)

# 是否优先使用已有模型
use_existing_model = False   # 可以自行更改，True 表示使用已有模型，False 表示重新训练模型
if not (use_existing_model and os.path.exists(model_path)):
    print("Training KMeans model...")
    # 对图像进行 KMeans 聚类
    images_kmeans_train(images_train, k1=2, batch_size=512*512*100, 
                                    max_iter=100, tol=1e-4, random_state=42, 
                                    model_path=model_path)
else:
    print("Using existing model...")
    
# 预测图像的分割结果
labels = images_kmeans_predict(images_test, model_path=model_path)

# 统一标签，防止颜色翻转
final_labels = align_labels(labels)

# 1. 形态学开运算去噪
labels_denoised = remove_noise_morphology(final_labels, operation="MORPH_CLOSE", kernel_size=3)
labels_denoised = remove_noise_morphology(labels_denoised, operation="MORPH_OPEN", kernel_size=3)

# 2. 连通区域分析去小噪声
labels_denoised = remove_noise_connected_components(labels_denoised, min_size=500)

# ============= **分页显示结果** =============
images_per_page = 10  # 每页显示 10 张
num_images = len(labels_denoised)
num_pages = int(np.ceil(num_images / images_per_page))  # 计算总页数

for page in range(num_pages):
    start_idx = page * images_per_page
    end_idx = min(start_idx + images_per_page, num_images)
    
    plt.figure(figsize=(15, 5 * (end_idx - start_idx)))  # 调整图像大小
    for i, idx in enumerate(range(start_idx, end_idx)):
        plt.subplot(end_idx - start_idx, 3, i * 3 + 1)
        plt.imshow(images_test[idx], cmap='gray')
        plt.title(f'Original Image {idx+1}')
        plt.axis('off')

        plt.subplot(end_idx - start_idx, 3, i * 3 + 2)
        plt.imshow(test_masks[idx], vmin=0, vmax=1)
        plt.title(f'Ground Truth Mask {idx+1}')
        plt.axis('off')

        plt.subplot(end_idx - start_idx, 3, i * 3 + 3)
        plt.imshow(labels_denoised[idx], vmin=0, vmax=1)
        plt.title(f'Predicted Mask {idx+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()