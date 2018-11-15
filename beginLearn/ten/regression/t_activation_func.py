import tensorflow as tf
import numpy as np


def add_layer(input, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output


# 一个特性,300行
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None, 1])  # None表示给多少个例子都可以,1表示一个特征
ys = tf.placeholder(tf.float32,[None, 1])

lay = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
predition = add_layer(lay, 10, 1, activation_function=None)  # 第二层输出的预测值

# 对每一个例子进行求和 ，再取平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                                    reduction_indices=[1]))
# 减小误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 10:
            print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))
