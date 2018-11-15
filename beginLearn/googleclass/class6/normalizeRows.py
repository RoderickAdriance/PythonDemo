import numpy as np

def normalizeRows(x):
    #向量的模
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    print(norm)
    x = x / norm
    print(x)
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])

normalizeRows(x)
