import numpy as np

### One reason why we use "numpy" instead of "math" in Deep Learning ###
# x = np.array([1,2,3])
# exp = np.exp(x)
# print(exp)

#矩阵的sigmoid函数
def sigmoid_vector(X):
    exp = 1 / (1+np.exp(-X))
    print(exp)


X = np.array([1,2,3])
vector = sigmoid_vector(X)

