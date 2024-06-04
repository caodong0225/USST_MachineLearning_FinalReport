"""
使用MLP模型训练数据
"""
import numpy as np
from keras import Sequential
from keras.layers import BatchNormalization, Dense
from keras.optimizers import Adam
from processData.dataset import X_train, X_test, y_train, y_test


# 二分类的多标签准确度计算函数
def custom_accuracy(y_true, y_pred):
    """
    计算多标签分类的准确率
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 计算每个样本的准确率
    accuracies = (y_true == y_pred).mean(axis=1)
    # 返回所有样本的平均准确率
    return accuracies.mean()


def create_model():
    """
    创建MLP模型
    :return:
    """
    # 定义模型
    model_build = Sequential()
    # 输入层和隐藏层
    model_build.add(Dense(512, input_dim=425, activation='relu'))
    model_build.add(BatchNormalization())
    model_build.add(Dense(256, activation='relu'))
    model_build.add(Dense(128, activation='relu'))
    model_build.add(Dense(64, activation='relu'))
    model_build.add(BatchNormalization())
    # 输出层
    model_build.add(Dense(4, activation='sigmoid'))
    return model_build


model = create_model()

# 定义优化器
optimizer = Adam(learning_rate=0.001)
# 编译模型
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 打印模型摘要
model.summary()

# 训练模型
HISTORY = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test))

# 预测
mlp_pred_train = model.predict(X_train)
mlp_pred_test = model.predict(X_test)

# 将预测结果转化为二值（0或1）
mlp_pred_train_binary = (mlp_pred_train > 0.5).astype(int)
mlp_pred_test_binary = (mlp_pred_test > 0.5).astype(int)

# 计算二分类准确度
train_custom_accuracy = custom_accuracy(y_train, mlp_pred_train_binary)
test_custom_accuracy = custom_accuracy(y_test, mlp_pred_test_binary)
# 评估模型在训练集和测试集上的表现
train_loss, train_accuracy = model.evaluate(X_train, y_train)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("训练集得分：", train_accuracy)
print("测试集得分：", test_accuracy)
print("训练集二分类准确度：", train_custom_accuracy)
print("测试集二分类准确度：", test_custom_accuracy)
# 训练集得分： 0.44921875
# 测试集得分： 0.23046875
# 训练集二分类准确度： 0.981689453125
# 测试集二分类准确度： 0.5849609375
