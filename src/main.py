import numpy as np
import os
from core.train import images_kmeans_train
from core.predict import images_kmeans_predict
from core.eval import compute_iou
from util.data_visualization import display_images_paginated
from util.data_process import (
    normalize_ct,
    align_labels,
    remove_noise_morphology,
    remove_noise_connected_components,
    convert_mask_auto
)

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
train_image_path = os.path.join(current_dir, '../data/split', 'train_images.npy')
test_image_path = os.path.join(current_dir, '../data/split', 'test_images.npy')
test_mask_path = os.path.join(current_dir, '../data/split', 'test_masks.npy')
predict_mask_path = os.path.join(current_dir, '../data/submit', 'predict_masks.npy')
model_path = os.path.join(current_dir, '../model/', 'kmeans_model.pkl')

# 加载图像数据
num_images_to_process = 746
start_num = 0
images_train = np.load(train_image_path)[start_num : start_num + num_images_to_process]  
images_train = normalize_ct(images_train)
images_train = images_train.squeeze()  

images_test = np.load(test_image_path)
images_test = normalize_ct(images_test)
images_test = images_test.squeeze()

# 加载测试集的 mask
test_masks = np.load(test_mask_path)
test_labels = convert_mask_auto(test_masks)

# 是否优先使用已有模型
use_existing_model = True   # 可以自行更改，True 表示使用已有模型，False 表示重新训练模型
if not (use_existing_model and os.path.exists(model_path)):
    print("Training KMeans model...")
    # 对图像进行 KMeans 聚类
    images_kmeans_train(images=images_train, k1=2, batch_size=512*512*100, 
                        max_iter=500, tol=1e-4, random_state=42, 
                        verbose=True, model_path=model_path)
else:
    print("Using existing model...")
    
# 预测图像的分割结果
predict_labels = images_kmeans_predict(images_test, model_path=model_path)

# 统一标签，防止颜色翻转
predict_labels = align_labels(predict_labels)

# 形态学开运算去噪
predict_labels = remove_noise_morphology(predict_labels, operation="MORPH_CLOSE", kernel_size=3)
predict_labels = remove_noise_morphology(predict_labels, operation="MORPH_OPEN", kernel_size=3)

# 连通区域分析去小噪声
predict_labels = remove_noise_connected_components(predict_labels, min_size=500)

# 计算 IoU 指数
iou_scores = compute_iou(test_labels, predict_labels)
print("IoU 指数: ", iou_scores)

# 保存预测结果
if not os.path.exists(os.path.dirname(predict_mask_path)):
    os.makedirs(os.path.dirname(predict_mask_path))
    
predict_masks = convert_mask_auto(predict_labels)
np.save(predict_mask_path, predict_masks)

# 分页显示结果
display_images_paginated(images_test, test_labels, predict_labels, images_per_page=7)