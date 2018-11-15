import tensorflow as tf
import numpy as np

#开始端-1  结束端-1 分割300个数据,生成线段
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
#平方-0.5加上噪声
y_data = np.square(x_data) - 0.5 + noise
print(y_data)