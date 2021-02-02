import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#下载鸢尾花数据集
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

COLUMN_NAMES = ['SpalLength','SepalWidth','PetalLength','PetaWidth','Species']
df_iris = pd.read_csv(train_path,names=COLUMN_NAMES,header=0)
iris = np.array(df_iris)

'''
#单一图

plt.scatter(iris[:,2],iris[:,3],c=iris[:,4],cmap='brg') # c= cmap=
plt.title("Anderson's Iris Data set\n(Blue -> Setosa | Red -> Versicolor | Green -> Virginica)")
plt.xlabel(COLUMN_NAMES[2])
plt.ylabel(COLUMN_NAMES[3])
plt.show()

'''
'''
#有效组合 6钟
# 设置画布尺寸
fig = plt.figure('Iris Data', figsize=(15, 3))
# 设置整个的画布标题
fig.suptitle("Anderson's Iris Data Set\n(Blue->Setosa | Red->Versicolor | Green->Virginica)")

for i in range(4):
    plt.subplot(1, 4, i + 1)
    if i == 0:
        plt.text(0.3, 0.5, COLUMN_NAMES[0], fontsize=15)
    else:
        plt.scatter(iris[:, i], iris[:, 0], c=iris[:, 4], cmap='brg')
    plt.title(COLUMN_NAMES[i])
    plt.ylabel(COLUMN_NAMES[0])

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()'''

#所有 4*4 大图
# 设置画布尺寸
fig = plt.figure('Iris Data', figsize=(15, 15))
# 设置整个的画布标题
fig.suptitle("Anderson's Iris Data Set\n(Blue->Setosa | Red->Versicolor | Green->Virginica)")

for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 4 * i + (j + 1))
        if i == j:
            plt.text(0.3, 0.4, COLUMN_NAMES[i], fontsize=15)
        else:
            plt.scatter(iris[:, j], iris[:, i], c=iris[:, 4], cmap='brg')

        if i == 0:
            plt.title(COLUMN_NAMES[j])
        if j == 0:
            plt.ylabel(COLUMN_NAMES[i])

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()



