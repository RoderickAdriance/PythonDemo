import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    #随机变量,形状 in_size行 out_size列的随机矩阵
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))
    #1行 out_size列的0.1矩阵
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    #尚未被激活的值
    Wx_plus_b=tf.matmul(inputs,Weight)+biases

    if activation_function is None:
        outputs=Wx_plus_b
    else:
        #使用传进来的激励函数
        outputs=activation_function(Wx_plus_b)
    return outputs


#开始端-1  结束端-1 分割300个数据,1行300列
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
#平方-0.5加上噪声,y_data为一行300列的随机值
y_data = np.square(x_data) - 0.5 + noise

#利用占位符定义我们所需的神经网络的输入。 tf.placeholder()就是代表占位符，
# 这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#定义隐藏层，使用 Tensorflow 自带的激励函数tf.nn.relu
lay = add_layer(xs, 1, 10, activation_function=tf.nn.relu)#输入1列输出10列
#定义输出层,结果
prediction = add_layer(lay, 10, 1, activation_function=None)#输入10列输出1列

#计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

#tf.train.GradientDescentOptimizer()中的值通常都小于1。
# 这里取的是0.1，代表以0.1的效率来最小化误差loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#梯度优化

#变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练1000次
#机器学习的内容是train_step, 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    #每50步我们输出一下机器学习的误差。
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()#本次运行请注释，全局运行不要注释
#每隔50次训练刷新一次图形，用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.1s。
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)

plt.show()