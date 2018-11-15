import numpy as np

def softmax(x):
    e = np.exp(x)
    sum = np.sum(e,axis=1,keepdims=True)
    v = e / sum
    print(v)

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
softmax(x)