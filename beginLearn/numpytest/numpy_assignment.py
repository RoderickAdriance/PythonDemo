import numpy as np
a = np.arange(4)
# array([0, 1, 2, 3])
b = a
c = a
d = b

a[0] = 11   #改变a的第一个值，b、c、d的第一个值也会同时改变。
print(a)
# array([11,  1,  2,  3])
print(b is a)

# copy() 的赋值方式没有关联性
b = a.copy()    # deep copy
print(b)        # array([11, 22, 33,  3])
a[3] = 44
print(a)        # array([11, 22, 33, 44])
print(b)        # array([11, 22, 33,  3])