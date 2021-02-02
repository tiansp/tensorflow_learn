'''
import tensorflow as tf
import numpy as np

# x是一个长度为4的一维数组
x = np.array([1, 2, 3, 4])
# W和B都是数字
w = tf.Variable(1.)
b = tf.Variable(1.)

y = 1/(1+tf.exp(-(w*x+b)))
# 结果也是一个长度为4的一维张量
# <tf.Tensor: id=36, shape=(4,), dtype=float32, numpy=array([0.880797  , 0.95257413, 0.98201376, 0.9933072 ], dtype=float32)>

# y是样本标签
y = np.array([0, 0, 1, 1])
# pred是预测概率
pred = np.array([0.1, 0.2, 0.8, 0.49])
# `1-y`和`1-pred`做广播运算结果也是一维数组
1-y
# array([1, 1, 0, 0])
1-pred
# array([0.9 , 0.8 , 0.2 , 0.51])
-tf.reduce_sum(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))
# <tf.Tensor: id=61, shape=(), dtype=float64, numpy=1.2649975061637104>
-tf.reduce_mean(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))
# <tf.Tensor: id=73, shape=(), dtype=float64, numpy=0.3162493765409276>

tf.exp()    tensorflow中使用exp函数来实现e的x次方的运算 这个函数的参数要求是浮点数，否则会报错
tf.math.log() 在tensorflow中使用，math.log()函数实现以e为底的对数运算 

# y和pred都是一维数组，1-y和1-pred做广播运算结果也是一维数组。
# 
# y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred)这是每一个样本的交叉熵损失，对他们求和tf.reduce_sum(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))就可以得到所有样本的交叉熵损失，要记住前面还有一个负号。
# 
# 使用tf.reduce_mean()函数可以得到平均交叉熵损失-tf.reduce_mean(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))。


准确率是正确分类的样本数除以样本总数。

通过sigmoid()函数得到的预测值是一个概率，首先要把它转换为类别0或1。
# 准确率
# y是样本标签
y = np.array([0, 0, 1, 1])
# pred是预测概率
pred = np.array([0.1, 0.2, 0.8, 0.49])

"""如果将阈值设置为0.5，那么可以使用四舍五入函数round()的把它转化为0或1"""
tf.round(pred)
# <tf.Tensor: id=73, shape=(4,), dtype=float64, numpy=array([0., 0., 1., 0.])>

"""使用equal()函数逐元素的去比较预测值和标签值"""
tf.equal(tf.round(pred), y)
"""得到的结果是一个和y、pred形状相同的一维张量"""
"""可以看到其中前三对元素是相同的，最后一对元素不相同"""
# <tf.Tensor: id=77, shape=(4,), dtype=bool, numpy=array([ True,  True,  True, False])>

"""使用cast()函数把这个结果转化为整数"""
tf.cast(tf.equal(tf.round(pred), y), tf.int8)
# <tf.Tensor: id=82, shape=(4,), dtype=int8, numpy=array([1, 1, 1, 0], dtype=int8)>

"""对它的所有元素求平均值"""
tf.reduce_mean(tf.cast(tf.equal(tf.round(pred), y), tf.float32))
# <tf.Tensor: id=89, shape=(), dtype=float32, numpy=0.75>

"""如果参数恰好是0.5，那么返回的结果是0"""
tf.round(0.5)
# <tf.Tensor: id=91, shape=(), dtype=float32, numpy=0.0>

"""当参数的值大于0.50，结果才是1"""
tf.round(0.500001)
# <tf.Tensor: id=93, shape=(), dtype=float32, numpy=1.0>

如果将阈值设置为0.5，那么可以使用四舍五入函数round()的把它转化为0或1，然后使用equal()函数逐元素的去比较预测值和标签值，得到的结果是一个和y、pred形状相同的一维张量。

可以看到其中前三对元素是相同的，最后一对元素不相同。

下面使用cast()函数把这个结果转化为整数，然后对它的所有元素求平均值，就可以得到正确样本在所有样本中的比例。

要注意的是使用round()函数时，如果参数恰好是0.5，那么返回的结果是0，当参数的值大于0.50，结果才是1。

where(condition, a, b)  根据条件condition返回a或者B的值

pred = np.array([0.1, 0.2, 0.8, 0.49])
tf.where(pred < 0.5, 0, 1)
# <tf.Tensor: id=101, shape=(4,), dtype=int32, numpy=array([0, 0, 1, 0])>


pred = np.array([0.1, 0.2, 0.8, 0.49])
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
tf.where(pred < 0.5, a, b)
当pred中的元素小于0.5时就返回a中对应位置的元素，否则返回B中对应位置的元素
#<tf.Tensor: id=109, shape=(4,), dtype=int32, numpy=array([ 1,  2, 30,  4])>

tf.where(pred >= 0.5)
# <tf.Tensor: id=111, shape=(1, 1), dtype=int64, numpy=array([[2]], dtype=int64)>

y = np.array([0, 0, 1, 1])
pred = np.array([0.1, 0.2, 0.8, 0.49])
tf.reduce_mean(tf.cast(tf.equal(tf.where(pred < 0.5, 0, 1), y), tf.float32))
# <tf.Tensor: id=120, shape=(), dtype=float32, numpy=0.75>

'''

# 第1步加载数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.array([137.97, 104.50, 100.00, 126.32, 79.20, 99.00, 124.00, 114.00,
              106.69, 140.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
y = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

# plt.scatter(x, y)
# plt.show()

# 第2步数据处理
x_train = x - np.mean(x)
y_train = y

# plt.scatter(x_train, y_train)
# plt.show()

# 第3步设置超参数和显示间隔
learn_rate = 0.005
iter = 5
display_step = 1

# 第4步设置模型变量的初始值
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

x_ = range(-80, 80)
y_ = 1/(1+tf.exp(-(w*x_+b)))
# plt.plot(x_, y_)


# 可视化输出
# plt.scatter(x_train, y_train)
# plt.plot(x_, y_, color="red", linewidth=3)

# 第5步训练模型
cross_train = []
acc_train = []
for i in range(0, iter + 1):
    with tf.GradientTape() as tape:
        pred_train = 1/(1 + tf.exp(-(w*x_train+b)))
        Loss_train = -tf.reduce_mean(y_train*tf.math.log(pred_train)+(1-y_train)*tf.math.log(1-pred_train))
        Accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train < 0.5, 0, 1), y_train), tf.float32))

    cross_train.append(Loss_train)
    acc_train.append(Accuracy_train)

    dL_dw, dL_db = tape.gradient(Loss_train, [w, b])

    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    if i % display_step == 0:
        print("i: %i, Train Loss: %f, Accuracy: %f" % (i, Loss_train, Accuracy_train))
        y_ = 1/(1+tf.exp(-(w*x_+b)))
        # plt.plot(x_, y_)
# plt.show()

"""这是商品房面积"""
x_test = [128.15, 45.00, 141.43, 106.27, 99.00, 53.84, 85.36, 70.00, 162.00, 114.60]
"""根据面积计算概率，np.mean(x)这里使用训练数据的平均值，对新的数据进行中心化处理"""
pred_test = 1/(1+tf.exp(-(w*(x_test-np.mean(x))+b)))
"""根据概率进行分类"""
y_test = tf.where(pred_test<0.5, 0, 1)
for i in range(len(x_test)):
    print(x_test[i], "\t", pred_test[i].numpy(), "\t", y_test[i].numpy(), "\t")

plt.scatter(x_test, y_test)
x_ = np.array(range(-80, 80))
y_ = 1/(1+tf.exp(-(w*x_+b)))
"""因为散点图的X坐标使用的是真实的面积，没有平移，所以在这里加上训练级的均值np.mean(x)平移曲线"""
plt.plot(x_+np.mean(x), y_)
plt.show()












