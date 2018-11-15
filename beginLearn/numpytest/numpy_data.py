import numpy as np
# array：创建数组
# dtype：指定数据类型
# zeros：创建数据全为0
# ones：创建数据全为1
# empty：创建数据接近0
# arrange：按指定范围创建数据
# linspace：创建线段
a = np.array([2,23,4])  # list 1d
print(a)
# [2 23 4]

a = np.array([2,23,4],dtype=np.int)
print(a.dtype)
# int 64

a = np.array([2,23,4],dtype=np.float)
print(a.dtype)
# float64


a = np.array([[2,23,4],[2,32,4]])  # 2d 矩阵 2行3列
print(a)

a = np.zeros((3,4)) # 数据全为0，3行4列
print(a)

a = np.empty((3,4)) # 数据为empty，3行4列  每个值都是接近于零的数
print(a)

a = np.arange(10,20,2) # 10-19 的数据，2步长
print(a)

#reshape 改变数据的形状
a = np.arange(12)    # 3行4列，0到11
print(a)
a=a.reshape((3,4))
print(a)

#linspace 创建线段型数据
a = np.linspace(1,10,20)    # 开始端1，结束端10，且分割成20个数据，生成线段
print(a)

a = np.linspace(1,10,20).reshape((5,4)) # 更改shape
print(a)