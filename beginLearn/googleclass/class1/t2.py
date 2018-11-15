import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float', [None, 784])

W = tf.Variable(tf.zeros([784, 10]))  # 每个像素点,对应10个类别的各自权重
b = tf.Variable(tf.zeros([10]))  # 10各类别对应的偏置

y = tf.nn.softmax(tf.matmul(x, W) + b)  # 预测样本属于10个类别各自的概率
y_ = tf.placeholder("float", [None, 10]) #真实样本属于10个类别各自的概率

cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) #计算交叉熵

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#使用梯度下降最小化交叉熵

init=tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    #每次随机抓取100个图片
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})

# 最大值1所在的索引位置就是类别标签：[True, False, True, True]
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
