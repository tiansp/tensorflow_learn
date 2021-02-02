import numpy as np

# a = np.array([[1,2,3,4]])
# print(a)


# print(np.ones((3,2)))   (3,2) 参数为元组

# print(np.zeros((2,2)))

# print(np.eye(2,2)) 参数正常

#print(np.linspace(1,10,4)) 起始数 结束数 总共个数 等差数列

# print(np.logspace(1,10,10,base=2)) base=要加，等比数列

#asarray() 将列表或元组转换成数组，若原数组存在则直接使用，列表或元组改变会影响数组值
# list1 = [[1,1,1],[1,1,1],[1,1,1]]
# arr1 = np.array(list1)
# arr2 = np.asarray(list1)
#
# list1[0] = 0
#
# print("list1:\n",list1)
# print("arr1:\n",arr1)
# print("arr2:\n",arr2)

#结果：
'''
list1:
 [0, [1, 1, 1], [1, 1, 1]]
arr1:
 [[1 1 1]
 [1 1 1]
 [1 1 1]]
arr2:
 [[1 1 1]
 [1 1 1]
 [1 1 1]]
 '''


#数组运算
'''
b = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
b_1 = b[0]  第一行
b_2 = b[0:2]    前两行
b_3 = b[:2]     同上
b_4 = b[0:2,0:2]    前两行前两个  
b_5 = b[0:2,1:3]    第一行前两个，第二行23个
b_6 = b[:,0]    输出列
print('b_1: \n',b_1)        
print('b_2: \n',b_2)        
print('b_3: \n',b_3)
print('b_4: \n',b_4)
print('b_5: \n',b_5)
print('b_6: \n',b_6)
'''

'''
b_1: 
 [0 1 2 3]
b_2: 
 [[0 1 2 3]
 [4 5 6 7]]
b_3: 
 [[0 1 2 3]
 [4 5 6 7]]
b_4: 
 [[0 1]
 [4 5]]
b_5: 
 [[1 2]
 [5 6]]
b_6: 
 [0 4 8]
 '''

'''
b = np.arange(12)
print('原\n',b)
print('不改变原数组形状\n',b.reshape(3,4))
print('不改变原数组形状\n',b)
b.resize((3, 4))
print('改变原数组形状\n',b)

改变前后元素数应相同
可用于快速创建数组
值可为-1，系统自动计算行/列数

原
 [ 0  1  2  3  4  5  6  7  8  9 10 11]
不改变原数组形状
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
不改变原数组形状
 [ 0  1  2  3  4  5  6  7  8  9 10 11]
改变原数组形状
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

'''
'''
t = np.arange(24).reshape(2,3,4)
print(t)

[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

'''

#数组计算符合四则运算


#求和
#sum() axis= 轴，按轴求和
'''
t = np.arange(24).reshape((2,3,4))
print('t=\n',t)
print('sum=\n',np.sum(t,axis=0))

t=
 [[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
sum=
 [[12 14 16 18]
 [20 22 24 26]
 [28 30 32 34]]
'''
#np.prod() 乘积  np.diff() 相邻元素的差  np.sqrt() 各元素的平方根  np.exp() np.abs()


#数组堆叠运算 np.stack((a1,a2,...),axis) 按轴堆叠
'''
x = np.array([1,2,3])
y = np.array([4,5,6])
print(np.stack((x,y),axis=0))
print(np.stack((x,y),axis=1))

[[1 2 3]
 [4 5 6]]

[[1 4]
 [2 5]
 [3 6]]
 '''
#矩阵 np.matrix(str/list/tumlp/arr)

'''
print(np.mat('1 2 3 ; 4 5 6 '))
# [[1 2 3]
#  [4 5 6]]
print(np.mat([[1,2,3],[4,5,6]]))
# [[1 2 3]
#  [4 5 6]]

矩阵转置 .T
矩阵求逆 .I
'''


#随机数 np.random
'''
np.random.rand()        元素在0-1均匀分布，浮点数
np.random.uniform(a,b,size)     元素在a,b之间均匀分布，浮点数
np.random.randint(a,b,size)     元素在a,b之间均匀分布，整数
np.random.randn()       标准正态分布
np.random.normal()      正态分布
'''

#打乱顺序函数   np.random.shuffle()
