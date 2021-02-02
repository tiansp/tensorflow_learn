import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
TEST_URL = "http://download.tensorflow.org/data/iris_training.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

df_iris_train = pd.read_csv(train_path, header=0)
df_iris_test = pd.read_csv(test_path, header=0)
# 第2步处理数据
""""别读取训练集和测试集中的数据，并把它们转化为numpy数组"""
iris_train = np.array(df_iris_train)
iris_test = np.array(df_iris_test)
"""iris的训练集中有120条样本"""
print(iris_train.shape)
# (120, 5)
"""测试集中有30个样本"""
print(iris_test.shape)
# (30, 5)

"""每条样本中有5列，前4列是属性，最后一列是标签。
取出前两列属性花萼的长度和宽度来训练模型"""
train_x = iris_train[:, 0:2]
test_x = iris_test[:, 0:2]
"""取出最后一列作为标签值"""
train_y = iris_train[:, 4]
test_y = iris_test[:, 4]

print(train_x.shape, train_y.shape)
# (120, 2) (120,)
print(test_x.shape, test_y.shape)
# (30, 2) (30,)

"""从训练集中提取出标签值为0和1的样本，山鸢尾和变色鸢尾"""
x_train = train_x[train_y < 2]
y_train = train_y[train_y < 2]
"""训练集中有78条样本"""
print(x_train.shape, y_train.shape)
# (78, 2) (78,)

"""从测试集中提取出标签值为0和1的样本，山鸢尾和变色鸢尾"""
x_test = test_x[test_y < 2]
y_test = test_y[test_y < 2]
"""测试集中有22条样本"""
print(x_test.shape, y_test.shape)
# (22, 2) (22,)


"""分别记录训练集合测试集中的样本数"""
num_train = len(x_train)
num_test = len(x_test)
print(num_train, num_test)
# 78,22

# 按列中心化
x_train = x_train - np.mean(x_train, axis=0)
x_test = x_test - np.mean(x_test, axis=0)

print(np.mean(x_train, axis=0))
print(np.mean(x_test, axis=0))
# [-2.61898765e-16 -5.29490981e-16]
# [-2.61898765e-16 -5.29490981e-16]

plt.figure(figsize=(10, 3))
cm_pt = mpl.colors.ListedColormap(["blue", "red"])

plt.subplot(121)
"""绘制散点图"""
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)

plt.subplot(122)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_pt)
plt.show()

# 构造多元线性模型需要的属性矩阵和标签列向量。
x0_train = np.ones(num_train).reshape(-1, 1)
X_train = tf.cast(tf.concat((x0_train, x_train), axis=1), tf.float32)
Y_train = tf.cast(y_train.reshape(-1, 1), tf.float32)
print(X_train.shape, Y_train.shape)
# (22, 3)

x0_test = np.ones(num_test).reshape(-1, 1)
X_test = tf.cast(tf.concat((x0_test, x_test), axis=1), tf.float32)
Y_test = tf.cast(y_test.reshape(-1, 1), tf.float32)
print(X_test.shape, Y_test.shape)
# (22, 1)
# 第3步设置超参数，设置模型参数初始值
learn_rate = 0.2
iter = 120
display_step = 30

np.random.seed(612)
W = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)

# 第4步训练模型
"""列表ce用来保存每一次迭代的交叉熵损失"""
ce_train = []
"""acc用来保存准确率"""
acc_train = []
ce_test = []
acc_test = []
for i in range(0, iter + 1):
    with tf.GradientTape() as tape:
        """这是多元模型的sigmoid()函数
        属性矩阵X和参数向量W相乘的结果是一个列向量，
        因此计算得到的PRED的也是一个列向量，是每个样本的预测概率"""
        PRED_train = 1 / (1 + tf.exp(-tf.matmul(X_train, W)))
        """这是计算交叉熵损失"""
        """Y*tf.math.log(PRED)+(1-Y)*tf.math.log(1-PRED)
        这一部分的结果是一个列向量，是每个样本的损失
        使用reduce_mean()函数求它们的平均值，得到平均交叉熵损失"""
        LOSS_train = -tf.reduce_mean(Y_train * tf.math.log(PRED_train) + (1 - Y_train) * tf.math.log(1 - PRED_train))

        PRED_test = 1 / (1 + tf.exp(-tf.matmul(X_test, W)))
        LOSS_test = -tf.reduce_mean(Y_test * tf.math.log(PRED_test) + (1 - Y_test) * tf.math.log(1 - PRED_test))
    """这是准确率也是一个数字，因为不需要对它求导，所以把它放在with语句的外面"""
    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_train < 0.5, 0., 1.), Y_train), tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_test < 0.5, 0., 1.), Y_test), tf.float32))

    """记录每一次迭代的损失和准确率"""
    ce_train.append(LOSS_train)
    acc_train.append(accuracy_train)
    ce_test.append(LOSS_test)
    acc_test.append(accuracy_test)

    """只使用训练集来更新模型参数"""
    dL_dw = tape.gradient(LOSS_train, W)
    W.assign_sub(learn_rate * dL_dw)

    """输出准确率和损失"""
    if i % display_step == 0:
        print("i: %i, TrainAcc: %f, TrainLoss: %f, TestAcc: %f, TestLoss: %f" % (i, accuracy_train, LOSS_train,
                                                                                 accuracy_test, LOSS_test))

# 第5步可视化
plt.figure(figsize=(10, 3))
plt.subplot(121)
plt.plot(ce_train, color="blue", label="train")
plt.plot(ce_test, color="red", label="test")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
plt.plot(acc_train, color="blue", label="train")
plt.plot(acc_test, color="red", label="test")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

'''
i: 0, TrainAcc: 0.230769, TrainLoss: 0.994269, TestAcc: 0.230769, TestLoss: 0.994269
i: 30, TrainAcc: 0.961538, TrainLoss: 0.481893, TestAcc: 0.961538, TestLoss: 0.481893
i: 60, TrainAcc: 0.987179, TrainLoss: 0.319128, TestAcc: 0.987179, TestLoss: 0.319128
i: 90, TrainAcc: 0.987179, TrainLoss: 0.246626, TestAcc: 0.987179, TestLoss: 0.246626
i: 120, TrainAcc: 1.000000, TrainLoss: 0.204982, TestAcc: 1.000000, TestLoss: 0.204982
'''