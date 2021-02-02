import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.compat.v1.enable_eager_execution()

#加载数据集
bosten_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y) = bosten_housing.load_data()

num_train = len(train_x)
num_test = len(test_x) #后面创建全1矩阵需要

#数据处理

#数据归一化处理
x_train = (train_x - train_x.min(axis=0))/(train_x.max(axis=0) - train_x.min(axis=0))
y_train = train_y

x_test = (test_x - test_x.min(axis=0))/(test_x.max(axis=0) - test_x.min(axis=0))
y_test = test_y

x0_train = np.ones(num_train).reshape(-1,1)     #全1数组
x0_test = np.ones(num_test).reshape(-1,1)

X_train = tf.cast(tf.concat([x0_train, x_train], axis=1), tf.float32)
X_test = tf.cast(tf.concat([x0_test, x_test], axis=1), tf.float32)

Y_train = tf.constant(y_train.reshape(-1,1),tf.float32)
Y_test = tf.constant(y_test.reshape(-1,1),tf.float32)

#超参数
learn_rate = 0.01
iter = 2000
display_step = 200

#初始值
np.random.seed(612)
W = tf.Variable(np.random.randn(14,1),dtype=tf.float32)


#训练模型

mse_train = []
mse_test = []
for i in range(0,iter+1):

    with tf.GradientTape() as tape:

        pred_train = tf.matmul(X_train,W)
        loss_train = 0.5 * tf.reduce_mean(tf.square(Y_train - pred_train))

        pred_test = tf.matmul(X_test,W)
        loss_test = 0.5 * tf.reduce_mean(tf.square(Y_test - pred_test))

    mse_train.append(loss_train)
    mse_test.append(loss_test)

    dl_dw = tape.gradient(loss_train,W)
    W.assign_sub(learn_rate*dl_dw)

    if i % display_step == 0:
        print("i: %i,train loss: %f,test loss:%f"%(i,loss_train,loss_test))

#可视化
plt.rcParams["font.sans-serif"] = "SimHei"
plt.figure(figsize=(20, 4))

plt.subplot(131)
plt.ylabel("MSE")
plt.plot(mse_train, color="blue", linewidth=3)
plt.plot(mse_test, color="red", linewidth=1.5)

plt.subplot(132)
plt.plot(y_train, color="blue", marker="o", label="true_price")
plt.plot(pred_train, color="red", marker=".", label="predict")
plt.legend()
plt.ylabel("Price")

plt.subplot(133)
plt.plot(y_test, color="blue", marker="o", label="true_price")
plt.plot(pred_test, color="red", marker=".", label="predict" )
plt.legend()
plt.ylabel("Price")

plt.suptitle("多元—波士顿房价预测",fontsize= 20)

plt.show()


'''
i: 0,train loss: 263.193481,test loss:276.994110
i: 200,train loss: 36.176552,test loss:37.562954
i: 400,train loss: 28.789459,test loss:28.952513
i: 600,train loss: 25.520697,test loss:25.333916
i: 800,train loss: 23.460527,test loss:23.340538
i: 1000,train loss: 21.887274,test loss:22.039745
i: 1200,train loss: 20.596283,test loss:21.124842
i: 1400,train loss: 19.510202,test loss:20.467237
i: 1600,train loss: 18.587011,test loss:19.997719
i: 1800,train loss: 17.797461,test loss:19.671593
i: 2000,train loss: 17.118927,test loss:19.456858
'''












































