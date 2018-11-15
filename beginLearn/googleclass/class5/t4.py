import numpy as np

A= np.array([[56.0,0.0,4.4,68.0],[1.2,104,52.0,8.0],[1.8,135.0,99.0,0.9]])

#axis 0是列,  1是行
cal = A.sum(axis=0)
print(cal)

#除法使用方法与乘法类似
precentage = (A/cal)*100
print(precentage)

