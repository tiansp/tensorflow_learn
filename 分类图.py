import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

df_iris_train = pd.read_csv(train_path, header=0)
df_iris_test = pd.read_csv(test_path, header=0)
# 第2步处理数据
""""别读取训练集和测试集中的数据，并把它们转化为numpy数组"""
iris_train = np.array(df_iris_train)
iris_test = np.array(df_iris_test)
"""iris的训练集中有120条样本"""
# (120, 5)
"""可以看到测试集中有30个样本"""
# (30, 5)

"""每条样本中有5列，前4列是属性，最后一列是标签。
我们只取出前两列属性花萼的长度和宽度来训练模型"""
train_x = iris_train[:, 0:2]
test_x = iris_test[:, 0:2]
"""取出最后一列作为标签值"""
train_y = iris_train[:, 4]
test_y = iris_test[:, 4]

# (120, 2) (120,)
# (30, 2) (30,)

"""从训练集中提取出标签值为0和1的样本，也就是山鸢尾和变色鸢尾"""
x_train = train_x[train_y < 2]
y_train = train_y[train_y < 2]
"""可以训练集中有78条样本"""
# (78, 2) (78,)

"""从测试集中提取出标签值为0和1的样本，也就是山鸢尾和变色鸢尾"""
x_test = test_x[test_y < 2]
y_test = test_y[test_y < 2]
"""可以测试集中有22条样本"""
# (22, 2) (22,)


"""分别记录训练集合测试集中的样本数"""
num_train = len(x_train)
num_test = len(x_test)
# 78,22

# 按列中心化
x_train = x_train - np.mean(x_train, axis=0)
x_test = x_test - np.mean(x_test, axis=0)

# 构造多元线性模型需要的属性矩阵和标签列向量。
x0_train = np.ones(num_train).reshape(-1, 1)

X_train = tf.cast(tf.concat((x0_train, x_train), axis=1), tf.float32)
Y_train = tf.cast(y_train.reshape(-1, 1), tf.float32)

x0_test = np.ones(num_test).reshape(-1, 1)

X_test = tf.cast(tf.concat((x0_test, x_test), axis=1), tf.float32)
Y_test = tf.cast(y_test.reshape(-1, 1), tf.float32)


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

M = 300
"""首先根据鸢尾花花萼长度和花萼宽度取值范围，确定坐标范围。找到每一列中的最小值和最大值"""
x1_min, x2_min = x_train.min(axis=0)
x1_max, x2_max = x_train.max(axis=0)
"""使用花萼长度的取值范围作为横坐标的范围，使用花萼宽度的取值范围作为纵坐标的范围"""
t1 = np.linspace(x1_min, x1_max, M)
t2 = np.linspace(x2_min, x2_max, M)
"""使用它们生成网格点坐标矩阵"""
m1, m2 = np.meshgrid(t1, t2)

m0 = np.ones(M * M)
"""生成多元线性模型需要的属性矩阵"""
X_mesh = tf.cast(np.stack((m0, m1.reshape(-1), m2.reshape(-1)), axis=1), dtype=tf.float32)
"""使用训练得到的模型参数w根据sigmoid公式计算所有网格点对应的函数值"""
Y_mesh = tf.cast(1 / (1 + tf.exp(-tf.matmul(X_mesh, W))), dtype=tf.float32)
"""把它们转化为分类结果0和1作为填充粉色还是绿色的依据"""
Y_mesh = tf.where(Y_mesh < 0.5, 0, 1)
"""并对他进行维度变换，让他和m1，m2具有相同的形状，这是`pcolormesh()`函数对参数的要求"""
n = tf.reshape(Y_mesh, m1.shape)

"""定义绘制散点的颜色方案"""
cm_pt = mpl.colors.ListedColormap(["blue", "red"])
"""定义背景颜色方案"""
cm_bg = mpl.colors.ListedColormap(["#FFA0A0", "#A0FFA0"])

"""绘制分区图"""
plt.pcolormesh(m1, m2, n, cmap=cm_bg)
"""绘制散点图"""
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_pt)
plt.show()

M = 300
"""首先根据鸢尾花花萼长度和花萼宽度取值范围，确定坐标范围。找到每一列中的最小值和最大值"""
x1_min, x2_min = x_test.min(axis=0)
x1_max, x2_max = x_test.max(axis=0)
"""使用花萼长度的取值范围作为横坐标的范围，使用花萼宽度的取值范围作为纵坐标的范围"""
t1 = np.linspace(x1_min, x1_max, M)
t2 = np.linspace(x2_min, x2_max, M)
"""使用它们生成网格点坐标矩阵"""
m1, m2 = np.meshgrid(t1, t2)

m0 = np.ones(M * M)
"""生成多元线性模型需要的属性矩阵"""
X_mesh = tf.cast(np.stack((m0, m1.reshape(-1), m2.reshape(-1)), axis=1), dtype=tf.float32)
"""使用训练得到的模型参数w根据sigmoid公式计算所有网格点对应的函数值"""
Y_mesh = tf.cast(1 / (1 + tf.exp(-tf.matmul(X_mesh, W))), dtype=tf.float32)
"""把它们转化为分类结果0和1作为填充粉色还是绿色的依据"""
Y_mesh = tf.where(Y_mesh < 0.5, 0, 1)
"""并对他进行维度变换，让他和m1，m2具有相同的形状，这是`pcolormesh()`函数对参数的要求"""
n = tf.reshape(Y_mesh, m1.shape)

"""定义绘制散点的颜色方案"""
cm_pt = mpl.colors.ListedColormap(["blue", "red"])
"""定义背景颜色方案"""
cm_bg = mpl.colors.ListedColormap(["#FFA0A0", "#A0FFA0"])

"""绘制分区图"""
plt.pcolormesh(m1, m2, n, cmap=cm_bg)
"""绘制散点图"""
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_pt)
plt.show()
