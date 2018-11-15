import numpy as np
A = np.array([1, 1, 1])
B = np.array([2, 2, 2])

print(np.vstack((A, B)))  # vertical stack
"""
[[1,1,1]
 [2,2,2]]
"""

C = np.vstack((A,B))
print(A.shape,C.shape)
# (3,) (2,3)


#左右合并
D = np.hstack((A,B))       # horizontal stack

print(D)
# [1,1,1,2,2,2]\
print(A.shape,D.shape)
# (3,) (6,)



#A = np.array([1, 1, 1])
# B = np.array([2, 2, 2])
#转置
print(A[np.newaxis,:])
# [[1 1 1]]

print(A[np.newaxis,:].shape)
# (1,3)

print(A[:,np.newaxis])
"""
[[1]
[1]
[1]]
"""

print(A[:,np.newaxis].shape)
# (3,1)
