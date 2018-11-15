import numpy as np

def L2(yhat,y):
    np_sum = np.sum((y - yhat) ** 2)
    return np_sum


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))