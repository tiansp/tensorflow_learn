import matplotlib.pyplot as plt
import numpy as np
'''
plt.rcParams["font.sans-serif"] = "SimHei"   #设置默认字体

# plt.rcParams()        恢复默认

plt.figure(facecolor='lightgrey')


# plt.plot()

plt.subplot(2,2,1)
plt.title("子标题1")
plt.subplot(2,2,2)
plt.title("子标题2",loc = 'left' , color = 'b')
plt.subplot(2,2,3)
myfontdict = {'fontsize':12,'color':'g','rotation':30}
plt.title("子标题3",fontdict=myfontdict)
plt.subplot(2,2,4)
plt.title("子标题4",color = 'w' , backgroundcolor = 'black')

plt.suptitle("全局标题",fontsize = 20 ,color = 'r' ,backgroundcolor = "y" )

plt.tight_layout(rect=[0,0,1,0.9])      #自动调整,参数为列表 [0,0,1,0.9]

plt.show()
'''
'''
#散点图

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False#设置正常显示负号

n = 1024
x1 = np.random.normal(0,1,n)#生成坐标
y1 = np.random.normal(0,1,n)

x2 = np.random.uniform(-4,4,(1,n))
y2 = np.random.uniform(-4,4,(1,n))

plt.scatter( x1 , y1 ,color = 'b', marker= '*',label = '正态分布')#绘制坐标点
plt.scatter( x2 , y2 ,color = 'y', marker= 'o',label = '均匀分布')

plt.title("标准正态分布",fontsize = 20)#标题
plt.legend()#图例显示
# plt.text(2.5,2.5,'均 值：0\n标准差：1')#文本

plt.xlim(-4,4)#坐标范围
plt.ylim(-4,4)

plt.xlabel("横坐标x",fontsize = 14)#坐标文本
plt.ylabel("纵坐标y",fontsize = 14)

plt.show()
'''


'''
#折线图

plt.rcParams["font.sans-serif"] = "SimHei"

n = 24
x1 = np.random.randint(27,37,n)
x2 = np.random.randint(40,68,n)

plt.plot(x1,label= '温度')
plt.plot(x2,label = '湿度')

plt.xlim(0,23)
plt.ylim(20,70)
plt.xlabel("小时",fontsize = 14)#坐标文本
plt.ylabel("测量值",fontsize = 14)

plt.title("24小时温度湿度测量值",fontsize = 20)#标题

plt.legend()#图例显示
plt.show()
'''

#柱状图
'''
plt.rcParams["font.sans-serif"] = "SimHei"

x1 = [32,25,16,30,24,45,40,33,28,17,24,20]
x2 = [-23,-35,-26,-35,-45,-43,-35,-32,-23,-17,-22,-28]

plt.bar(range(len(x1)),x1,width= 0.8 ,facecolor = 'g' ,edgecolor= 'white',label= '统计量1')
plt.bar(range(len(x2)),x2,width= 0.8 ,facecolor = 'r' ,edgecolor= 'white',label= '统计量2')

plt.title("柱状图",fontsize = 20)#标题
plt.legend()#图例显示
plt.show()
'''











