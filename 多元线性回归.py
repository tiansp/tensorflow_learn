import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.enable_eager_execution()
'''
#加载数据集
area = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
room = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])

#线性归一化
x1 = (area-area.min())/(area.max()-area.min())
x2 = (room-room.min())/(room.max()-room.min())
print(x1)
print(x2)
'''

#加载数据集
plt.rcParams['font.sans-serif'] = ['SimHei']

area = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
room = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
price = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
                  62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

num = len(area)

#数据处理
x0 = np.ones(num)

x1 = (area - area.min()) / (area.max() - area.min())
x2 = (room - room.min()) / (room.max() - room.min())

X = np.stack((x0, x1, x2), axis=1)
Y = price.reshape(-1, 1)

print(X.shape, Y.shape)
# ((16, 3), (16, 1))

#初始值
np.random.seed(612)
W = np.random.randn(3, 1)
W = tf.Variable(np.random.randn(3,1))
#超参数
learn_rate = 0.2      #学习率
iter = 50             #迭代次数
display_step = 10       #显示

#训练模型
'''
#numpy 实现
mse = []
for i in range(0, iter + 1):
    dl_dW = np.matmul(np.transpose(X), np.matmul(X, W) - Y)
    W = W - learn_rate * dl_dW

    pred = np.matmul(X, W)
    Loss = np.mean(np.square(Y - pred)) / 2

    mse.append(Loss)

    if i % display_step == 0:
        print("i: %i, Loss: %f" % (i, mse[i]))
'''
'''
(16, 3) (16, 1)
i: 0, Loss: 4229.356667
i: 50, Loss: 401.211563
i: 100, Loss: 106.858527
i: 150, Loss: 83.896398
i: 200, Loss: 81.843235
i: 250, Loss: 81.453758
i: 300, Loss: 81.238435
i: 350, Loss: 81.073364
i: 400, Loss: 80.941452
i: 450, Loss: 80.835367
i: 500, Loss: 80.749767
'''

#tensorfiow 自动求导
mse = []
for i in range(0,iter + 1):
    with tf.GradientTape() as tape:
        pred = tf.matmul(X,W)#张量相乘
        loss = tf.reduce_mean(tf.square(Y - pred)) / 2 #损失函数
    mse.append(loss)

    dL_dW = tape.gradient(loss, W) #求导
    W.assign_sub(learn_rate * dL_dW) #更新损失函数

    if i % display_step == 0:
        print("i: %i, Loss: %f "% (i, loss))
        
'''
(16, 3) (16, 1)
i: 0, Loss: 4551.271973, w: 0.209844, b: 0.404720
i: 50, Loss: 3959.990234, w: 3.405012, b: -3.985651
i: 100, Loss: 3450.553711, w: 6.369848, b: -8.061553
i: 150, Loss: 3011.633545, w: 9.120892, b: -11.845556
i: 200, Loss: 2633.468018, w: 11.673489, b: -15.358614
i: 250, Loss: 2307.648193, w: 14.041899, b: -18.620171
i: 300, Loss: 2026.927368, w: 16.239346, b: -21.648283
i: 350, Loss: 1785.063843, w: 18.278103, b: -24.459700
i: 400, Loss: 1576.679199, w: 20.169561, b: -27.069975
i: 450, Loss: 1397.138306, w: 21.924318, b: -29.493544
i: 500, Loss: 1242.448486, w: 23.552185, b: -31.743814
'''

print(tf.reshape(pred,[-1]).numpy)
#可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(mse)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.subplot(1, 2, 2)
PRED = tf.reshape(pred,[-1]).numpy()
plt.plot(price, color="red", marker="o", label="销售记录")
plt.plot(PRED, color="blue", marker=".", label="预测房价")
plt.xlabel("Sample", fontsize=14)
plt.ylabel("Price", fontsize=14)

plt.legend()
plt.show()