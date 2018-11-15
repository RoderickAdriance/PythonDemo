import numpy as np

A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]

C = np.vstack((A, B))  # vertical stack
D = np.hstack((A, B))  # horizontal stack

print(D)
"""
[[1 2]
[1 2]
[1 2]]
"""

print(A.shape, D.shape)
# (3,1) (3,2)


#axis参数很好的控制了矩阵的纵向或是横向打印
C = np.concatenate((A,B,B,A),axis=0)

print(C)

D = np.concatenate((A,B,B,A),axis=1)
print(D)
"""
array([[1, 2, 2, 1],
       [1, 2, 2, 1],
       [1, 2, 2, 1]])
"""