"""
加载数据集和标签
"""
import glob
from scipy.io import loadmat
import numpy as np


def load_dataset(path):
    """
    :param path: 文件路径
    :return: 返回数据集
    """
    files = glob.glob(path + "/*.mat")
    dataset_res = []
    for file in files:
        mat_data = loadmat(file)
        dataset_res.append(mat_data['x'])
    return np.array(dataset_res)


def load_labels(path):
    """
    :param path: 文件路径
    :return: 返回标签
    """
    labels_target = ["g_valence", "g_arousal", "g_dominance", "g_liking"]
    files = glob.glob(path + "/*.mat")
    labels_set = []
    for file in files:
        mat_data = loadmat(file)
        labels_set.append(np.array([mat_data[label] for label in labels_target]))
    labels_set = np.array(labels_set)
    # 去除最后一列的1维数据
    labels_set = np.squeeze(labels_set, axis=3)
    # 将第1维度和第2维度交换
    labels_set = np.swapaxes(labels_set, 1, 2)
    # 将labels_set转化为0-1标签
    labels_set = labels_set - 1
    return labels_set


if __name__ == '__main__':
    # 读取数据集
    dataset = load_dataset('../dataset/features')
    print(dataset.shape)
    # 读取标签
    labels = load_labels('../dataset/labels')
    print(labels.shape)
