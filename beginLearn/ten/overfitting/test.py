import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#设置随机函数的seed参数，对应的变量可以跨session生成相同的随机数：
tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01

# training data
x = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis] #[:, np.newaxis] 表示转置

y = x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]#np.random.randn随机20行数据

# test data
test_x = x.copy()
test_y = test_x + 0.3*np.random.randn(N_SAMPLES)[:, np.newaxis]

# show data
plt.scatter(x, y, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

