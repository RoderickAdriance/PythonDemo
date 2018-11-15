import numpy as np
import tensorflow as tf
x_data = np.random.rand(100).astype(np.float32)
print(x_data)
# # 随机变量,形状 in_size行 out_size列的随机矩阵
# Weight = tf.Variable(tf.random_normal([1, 10]))
# print("Weight:",Weight)
#
# Weight2 = tf.Variable(tf.random_normal([10, 1]))
# print("Weight2",Weight2)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# print(sess.run(Weight))
# print(sess.run(Weight2))
