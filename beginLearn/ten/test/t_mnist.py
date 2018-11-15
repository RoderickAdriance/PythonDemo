from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#下载测试数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None,10])  #真实值

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,W)+b)#定义函数,获得预测值
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #交叉熵loss函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#优化函数

init = tf.global_variables_initializer()
#对象在某一维上的其数据最大值所在的索引值，这里返回一个布尔数组。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#将布尔值转换为浮点数来代表对、错，然后取平均值。例如：[True, False, True, True]变为[1,0,1,1]，计算出平均值为0.75。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print(result)

