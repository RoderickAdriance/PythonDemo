import tensorflow as tf
import numpy as np

# 随机100个数字的数组
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensflow structure start

# 一维, 初始值-1 到1之间
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 一维，初始值 0.1
biases = tf.Variable(tf.zeros([1])+0.1)

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
# end tensflow structure start

sess=tf.Session()
sess.run(init)
#创建会话 Session. 用 Session 来执行 init 初始化步骤.
# 并且, 用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.
for step in range(201):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(Weights), sess.run(biases))
