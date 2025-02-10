import os
import numpy as np
from sklearn.model_selection import train_test_split

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 如果路径不存在，则创建data/origin目录
if not os.path.exists(os.path.join(current_dir, '../data/origin')):
    os.makedirs(os.path.join(current_dir, '../data/origin'), exist_ok=True)

# 数据路径
image_file_path = os.path.join(current_dir, '../data/origin', 'images_radiopedia.npy')
mask_file_path = os.path.join(current_dir, '../data/origin', 'masks_radiopedia.npy')

# 检测路径是否存在
assert os.path.exists(image_file_path), "图片数据不存在，请将images_radiopedia.npy放入data/origin目录！"
assert os.path.exists(mask_file_path), "掩码数据不存在，请将masks_radiopedia.npy放入data/origin目录！"

# 加载数据
images = np.load(image_file_path)  # shape: (829, ...)
masks = np.load(mask_file_path)  # shape: (829, ...)

# 检查数据是否匹配
assert images.shape[0] == masks.shape[0], "图片和掩码数量不匹配！"

# 随机划分数据集 (746 训练, 83 测试)
train_images, test_images, train_masks, test_masks = train_test_split(
    images, masks, test_size=83, random_state=425217476, shuffle=True
)

# 保存划分后的数据
output_dir = os.path.join(current_dir, '../data/split')  # 目标目录
os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）

np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
np.save(os.path.join(output_dir, 'test_images.npy'), test_images)
np.save(os.path.join(output_dir, 'test_masks.npy'), test_masks)

print(f"数据划分完成！\n训练集: {train_images.shape[0]} 张\n测试集: {test_images.shape[0]} 张")
