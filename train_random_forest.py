"""
训练随机森林
"""
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from processData.dataset import dataset, labels


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


# 载入数据
X = dataset
label_y = labels  # 这里直接使用所有的标签，不再选取某个通道的数据

# 升维
poly = preprocessing.PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X = preprocessing.scale(X)
X = preprocessing.normalize(X, norm='l1')
print(X.shape)

# 降维
pca = PCA(n_components=100)
X = pca.fit_transform(X)
print(X.shape)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, label_y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=5)

# 使用MultiOutputClassifier来进行多标签分类
multi_target_rf = MultiOutputClassifier(rf, n_jobs=-1)
multi_target_rf.fit(X_train, y_train)

# 预测
rf_pred_train = multi_target_rf.predict(X_train)
rf_pred_test = multi_target_rf.predict(X_test)

# 计算原有的训练集和测试集得分（子集准确率）
train_score = multi_target_rf.score(X_train, y_train)
test_score = multi_target_rf.score(X_test, y_test)

# 计算二分类准确度
train_custom_accuracy = custom_accuracy(y_train, rf_pred_train)
test_custom_accuracy = custom_accuracy(y_test, rf_pred_test)

print("训练集得分：", train_score)
print("测试集得分：", test_score)
print("训练集二分类准确度：", train_custom_accuracy)
print("测试集二分类准确度：", test_custom_accuracy)
# 训练集得分： 0.9990234375
# 测试集得分： 0.17578125
# 训练集二分类准确度： 0.999755859375
# 测试集二分类准确度： 0.630859375
