import numpy as np

# u = np.random.rand(10,2)
u = np.array([[1,2],[3,4],[5,6]])

v = np.array([[2],[3]])

a = u*v.transpose()
b= 1/a
print(a)
print(b)