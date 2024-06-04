"""
该文件实现了KNN模型的训练
"""
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import processData.dataset


# 二分类的多标签准确度计算函数
def custom_accuracy(y_true, y_pred):
    """
    计算多标签分类的准确率
    :param y_true:
    :param y_pred:
    :return:
    """
    # 计算每个样本的准确率
    accuracies = (y_true == y_pred).mean(axis=1)
    # 返回所有样本的平均准确率
    return accuracies.mean()


label_y = processData.dataset.labels
X = processData.dataset.dataset
# 升维
poly = sklearn.preprocessing.PolynomialFeatures(degree=2)  # 生成了二次多项式
X = poly.fit_transform(X)

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)  # 对数据进行缩放
# X=preprocessing.scale(X)
X = sklearn.preprocessing.normalize(X, norm='l1')  # L1正则化处理
print(X.shape)

pca = PCA(n_components=1000)
X = pca.fit_transform(X)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, label_y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
knn_pred_train = knn.predict(X_train)
knn_pred = knn.predict(X_test)

# 计算二分类准确度
train_custom_accuracy = custom_accuracy(y_train, knn_pred_train)
test_custom_accuracy = custom_accuracy(y_test, knn_pred)

print("训练集得分：", train_score)
print("测试集得分：", test_score)
print("训练集二分类准确度：", train_custom_accuracy)
print("测试集二分类准确度：", test_custom_accuracy)
# 训练集得分： 0.35833333333333334
# 测试集得分： 0.2125
# 训练集二分类准确度： 0.7294270833333333
# 测试集二分类准确度： 0.60625
