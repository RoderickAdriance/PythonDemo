import tensorflow as tf
import numpy as np
#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=0.1*x_data+0.3   #y=0.1x+0.3 直线方程


#Weights 实际值为0.1  biase为0.3
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))#1维  初始值 -1 到 1 的范围
biases = tf.Variable(tf.zeros([1]))#1维 初始值 0

y = Weights*x_data + biases
#计算 y 和 y_data 的误差
loss = tf.reduce_mean(tf.square(y-y_data))
#使用的误差传递方法是梯度下降法: Gradient Descent 使用 optimizer 来进行参数的更新
optimizer = tf.train.GradientDescentOptimizer(0.5)#0.5 学习效率
train = optimizer.minimize(loss)
### create tensorflow  structure end ###

#初始化所有之前定义的Variable
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)          # Very important

#创建会话 Session. 用 Session 来执行 init 初始化步骤.
# 并且, 用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.
for step in range(201):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(Weights), sess.run(biases))