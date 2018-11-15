import numpy as np
import tensorflow as tf

def auto_derivative():
    w = tf.Variable(0, dtype=tf.float32)
    # w = w^2 -10w +25
    # Jw:   2w-10=0   w=5导数为0点
    cost = w ** 2 - 10 * w + 25
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.global_variables_initializer()

    session = tf.Session()
    session.run(init)

    for i in range(1000):
        session.run(train)
        print(session.run(w))

#手动梯度下降求导
def manual_derivative(leaning_rate):
    w=0
    j = w ** 2 - 10 * w + 25
    for i in range(1000):
        dw = 2 * w - 10
        w = w - leaning_rate*dw
        print(w)

def input_coefficients():
    coefficients=np.array([[1.],[-20.],[10]])

    w=tf.Variable(0,dtype=tf.float32)
    x=tf.placeholder(tf.float32,[3,1])

    cost=w**2*x[0][0]+w*x[1][0]+x[2][0]
    train=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init=tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    print(session.run(w))

    for i in range(1000):
        session.run(train, feed_dict={x: coefficients})
        print(session.run(w))

input_coefficients()