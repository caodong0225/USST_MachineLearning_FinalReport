"""
数据集处理
"""
import numpy as np
from sklearn.model_selection import train_test_split
from processData.load_dataset import load_dataset, load_labels


# 读取数据集
dataset = load_dataset('./dataset/features')
# 读取标签
labels = load_labels('./dataset/labels')
# 压缩数据集
dataset = np.reshape(dataset, (1280, -1))
# 压缩标签
labels = np.reshape(labels, (1280, -1))
# 查看数据集和标签的形状
# print(dataset.shape)  # (1280, 425)
# print(labels.shape)  # (1280, 4)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
