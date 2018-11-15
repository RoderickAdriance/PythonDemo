import numpy as np

#两种写法
def sigmoid_grad(X):
    v = 1 / (1+np.exp(-X))
    v = np.exp(-X) / (1+np.exp(-X))**2
    print(v)

    v = 1 / (1 + np.exp(-X))
    dv = v * (1-v)
    print(dv)

X = np.array([1,2,3])
sigmoid_grad(X)