import numpy as np
#14 到 2
A = np.arange(14,2, -1).reshape((3,4))
print(A)

#最小值最大值则用于让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素。
# 并将这些指定的元素转换为最小值或者最大值。
print(np.clip(A,5,9))
# array([[ 9, 9, 9, 9]
#        [ 9, 9, 8, 7]
#        [ 6, 5, 5, 5]])



# print(np.sort(A)) #每行排序
#
# print(np.transpose(A))     #矩阵转置
# print(A.T)      #矩阵转置

