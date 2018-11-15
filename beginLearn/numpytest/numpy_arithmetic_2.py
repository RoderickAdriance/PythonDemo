import numpy as np
A = np.arange(2, 14).reshape((3, 4))
print(A)
# array([[ 2, 3, 4, 5]
#        [ 6, 7, 8, 9]
#        [10,11,12,13]])

#其中的 argmin() 和 argmax() 两个函数分别对应着求矩阵中最小元素和最大元素的索引。
print(np.argmin(A))  # 0
print(np.argmax(A))  # 11
#统计中的均值
print(np.mean(A))        # 7.5
print(np.average(A))     # 7.5

print(np.cumsum(A)) #累加运算函数

print(np.diff(A))   #累差运算函数

print(np.nonzero(A))    # (array([0,0,0,0,1,1,1,1,2,2,2,2]),array([0,1,2,3,0,1,2,3,0,1,2,3]))