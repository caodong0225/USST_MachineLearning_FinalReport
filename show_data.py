"""
数据集显示
"""
import matplotlib.pyplot as plt
import numpy as np
from processData.dataset import dataset
from processData.dataset import labels

# 可视化数据集
# 标签为四分类问题，分别为valence、arousal、dominance、liking


# print(dataset.shape)  # (1280, 425)
# print(labels.shape)  # (1280, 4)
# 热力图形式表示数据集
def plot_dataset_heatmap():
    """
    可视化数据集
    :return:
    """
    # 选取labels为[1,1,1,1]的数据
    # 可视化数据集
    data_negative = dataset[np.where(np.all(labels == [0, 0, 0, 0], axis=1))[0]]
    data_positive = dataset[np.where(np.all(labels == [1, 1, 1, 1], axis=1))[0]]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(data_negative[0].reshape(25, 17), aspect='auto', cmap='jet')
    plt.title('Negative')
    plt.axis('off')
    # 显示正样本
    plt.subplot(1, 2, 2)
    plt.imshow(data_positive[0].reshape(25, 17), aspect='auto', cmap='jet')
    plt.title('Positive')
    plt.axis('off')
    plt.show()


def plot_label():
    """
    可视化标签
    :return:
    """
    # 统计各类别的数量
    label_count = np.sum(labels, axis=0)
    print(label_count)
    # 可视化标签
    plt.bar(np.arange(4), label_count)
    plt.xticks(np.arange(4), ['valence', 'arousal', 'dominance', 'liking'])
    plt.ylabel('Number of samples')
    plt.title('Distribution of labels')
    plt.show()


def plot_accuracy():
    """
    可视化准确率
    :return:
    """
    accuracy_dict = {'SVM': {'train': 0.28125, 'test': 0.27734375, 'train_binary': 0.806689453125,
                             'test_binary': 0.62109375},
                     'KNN': {'train': 0.35833333333333334, 'test': 0.2125,
                             'train_binary': 0.67, 'test_binary': 0.63},
                     'RandomForest': {'train': 0.9990234375, 'test': 0.17578125
                         , 'train_binary': 0.999755859375, 'test_binary': 0.650859375},
                     'DecisionTree': {'train': 0.984375, 'test': 0.109375,
                                      'train_binary': 0.99609375, 'test_binary': 0.54815625},
                     'MLP': {'train': 0.44921875, 'test': 0.23046875, 'train_binary': 0.62,
                             'test_binary': 0.6}}

    # 可视化准确率
    plt.figure(figsize=(15, 5))

    # 绘制训练集和测试集准确率
    plt.subplot(1, 2, 1)
    models = list(accuracy_dict.keys())
    train_accuracy = [accuracy_dict[model]['train'] for model in models]
    test_accuracy = [accuracy_dict[model]['test'] for model in models]
    models_length = range(len(models))
    width = 0.35
    plt.bar(models_length, train_accuracy, width, label='Train')
    plt.bar([i + width for i in models_length], test_accuracy, width, label='Test')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy of different models')
    plt.xticks([i + width / 2 for i in models_length], models)
    plt.legend()

    # 绘制自定义准确率
    plt.subplot(1, 2, 2)
    train_binary_accuracy = [accuracy_dict[model]['train_binary'] for model in models]
    test_binary_accuracy = [accuracy_dict[model]['test_binary'] for model in models]
    plt.bar(models_length, train_binary_accuracy, width, label='Train binary')
    plt.bar([i + width for i in models_length], test_binary_accuracy, width, label='Test binary')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('binary Train and Test Accuracy of different models')
    plt.xticks([i + width / 2 for i in models_length], models)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    plot_dataset_heatmap()
    plot_label()
    plot_accuracy()
