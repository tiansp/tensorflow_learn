import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.enable_eager_execution()
#加载数据集
bosten_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y) = bosten_housing.load_data()

#数据处理
x_train = train_x[:,5]
y_train = train_y
x_test = test_x[:,5]
y_test = test_y

#超参数
learn_rate = 0.04
iter = 2000
display_step = 200

#初始值
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

#模型训练
mse_train = []
mse_test = []

for i in range(0,iter+1):

    with tf.GradientTape() as tape:

        pred_train = w * x_train + b
        loss_train = 0.5 * tf.reduce_mean(tf.square(y_train - pred_train))

        pred_test = w * x_test + b
        loss_test = 0.5 * tf.reduce_mean(tf.square(y_test - pred_test))

    mse_train.append(loss_train)
    mse_test.append(loss_test)

    dl_dw , dl_db = tape.gradient(loss_train,[w,b])

    w.assign_sub(learn_rate*dl_dw)
    b.assign_sub(learn_rate*dl_db)

    if i % display_step == 0:
        print("i: %i,train loss: %f,test loss:%f"%(i,loss_train,loss_test))


#可视化
plt.rcParams["font.sans-serif"] = "SimHei"
plt.figure(figsize=(15,10))

plt.subplot(221)
plt.scatter(x_train,y_train,color = 'b',label = 'data')
plt.plot(x_train,pred_train,color = 'r',label = 'model')
plt.legend(loc = 'upper left')

plt.subplot(222)
plt.plot(mse_train ,color = 'b',linewidth = 3,label = 'train loss')
plt.plot(mse_test,color = 'r',linewidth = 1.5,label = 'test loss')
plt.legend(loc = 'upper right')

plt.subplot(223)
plt.plot(y_train,color = 'b',marker = "o",label = 'train price')
plt.plot(pred_train,color = 'r',marker = ".",label = 'predict')
plt.legend()

plt.subplot(224)
plt.plot(y_test ,color = 'b',marker = "o",label = 'true price')
plt.plot(pred_test,color = 'r',marker = ".",label = 'predict') # tensor对象要转换成numpy数组 .numpy()
plt.legend()
plt.suptitle("波士顿房价预测",fontsize = 20)


plt.show()


'''
i: 0,train loss: 321.837585,test loss:337.568634
i: 200,train loss: 28.122616,test loss:26.237764
i: 400,train loss: 27.144739,test loss:25.099327
i: 600,train loss: 26.341949,test loss:24.141077
i: 800,train loss: 25.682899,test loss:23.332979
i: 1000,train loss: 25.141848,test loss:22.650162
i: 1200,train loss: 24.697670,test loss:22.072006
i: 1400,train loss: 24.333027,test loss:21.581432
i: 1600,train loss: 24.033667,test loss:21.164261
i: 1800,train loss: 23.787903,test loss:20.808695
i: 2000,train loss: 23.586145,test loss:20.504938
'''














