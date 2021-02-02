import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#加载波士顿房价数据集
boston_housing = tf.keras.datasets.boston_housing

(train_x ,train_y),(test_x,test_y) = boston_housing.load_data(test_split=0)#test_split 测试集与训练集的比例

# print("Training set",len(train_x))
# print("Test set",len(test_x))
#


#波士顿房价可视化
'''
plt.figure(figsize=(5,5))#画布大小
plt.scatter(train_x[:,5],train_y)#绘制散点图
plt.xlabel('RM')#x轴标签文本
plt.ylabel("Price($1000's)")
plt.title("5.RM-Price")
plt.show()
'''

'''
#波士顿各属性与房价关系
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False#设置正常显示负号

titles = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B-1000","LSTAT","MEDV"]

plt.figure(figsize=(12,12))

for i in range(13):
    plt.subplot(4,4,(i+1))
    plt.scatter(train_x[:,i],train_y)

    plt.xlabel(titles[i])
    plt.ylabel("Price($1000's)")
    plt.title(str(i+1)+"."+titles[i]+" - Price")

plt.tight_layout(rect=[0,0,1,0.95])
plt.suptitle("各个属性与房价的关系",fontsize= 20)
plt.show()
'''






