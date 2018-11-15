import numpy as np

#矩阵作用是舍弃for循环
a=np.arange(5)
# print("a=",a)
a +=5
# print("a+5=",a)

w=np.arange(3).reshape((3,1)) + np.arange(3)
# print(w)

a = np.arange(5).reshape(1,5)
print(a)
print(a.dot(a.transpose()))
