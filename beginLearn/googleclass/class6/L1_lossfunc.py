import numpy as np

def L1(yhat,y):
    np_sum = np.sum(np.abs(yhat - y))
    return np_sum

yhat = np.array([0.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

