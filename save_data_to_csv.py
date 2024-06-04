"""
将数据集和标签保存到csv文件
"""
import pandas as pd
from processData.dataset import dataset, labels


def save_dataset():
    """
    将数据集保存到csv文件
    :return:
    """
    train_data = dataset
    train_data = pd.DataFrame(train_data)
    train_data.to_csv("train_data.csv", index=False)


def save_labels():
    """
    将标签保存到csv文件
    :return:
    """
    labels_data = pd.DataFrame(labels)
    labels_data.to_csv("train_labels.csv", index=False)


if __name__ == "__main__":
    save_dataset()  # 保存数据集
    save_labels()  # 保存标签
