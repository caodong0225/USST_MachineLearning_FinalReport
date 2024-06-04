"""
读取数据
"""
import scipy.io
import numpy as np


def read_data(file_path):
    """
    :param file_path: 文件路径
    :return: 读取的数据
    """
    mat_data = scipy.io.loadmat(file_path)
    return mat_data['x']


def read_label(file_path):
    """
    :param file_path: 文件路径
    :return: 读取的标签
    """
    labels = ["g_valence", "g_arousal", "g_dominance", "g_liking"]
    mat_data = scipy.io.loadmat(file_path)
    return np.array([mat_data[label] for label in labels])


if __name__ == '__main__':
    # 读取数据
    data = read_data('../dataset/features/s01.mat')
    print(data, data.shape)
    # 读取标签
    label = read_label('../dataset/labels/s01.mat')
    print(label, label.shape)
