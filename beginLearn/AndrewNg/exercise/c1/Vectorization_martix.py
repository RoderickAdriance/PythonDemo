import numpy as np

#使用矩阵进行计算
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]


# dot = np.dot(x1,x2)
# print(dot)
#
# outer = np.outer(x1,x2)
# print(outer)

mul = np.multiply(x1,x2)
print(mul)

W = np.random.randn(3,len(x1))

dot = np.dot(W,x1)
print(dot)