import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from exercise.c10.tensortest.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


def linear_function():
    np.random.seed(1)   #固定一个正态分布值

    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.array(np.random.rand(4, 1)), name="b")
    Y = tf.add(tf.matmul(W, X), b)

    session = tf.Session()
    return session.run(Y)


def sigmoid(z):
    x = tf.placeholder(tf.float32, name="x")

    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})
    return result

def cost(y_hat,labels):
    z=tf.placeholder(tf.float32,name="z")
    y=tf.placeholder(tf.float32,name="y")

    #交叉熵损失
    cost=tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)

    session = tf.Session()

    cost=session.run(cost,feed_dict={z:y_hat,y:labels})
    return cost

def one_hot_matrix(labels,C):
    C=tf.constant(C,name="C")

    one_hot_matrix = tf.one_hot(labels,C,axis=0)

    sess = tf.Session()

    one_hot = sess.run(one_hot_matrix)

    return one_hot

def test_one_hot_matrix():
    labels = np.array([1, 2, 3, 0, 1, 2, 3])
    one_hot = one_hot_matrix(labels, C=4)
    print("one_hot:",one_hot)

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    print("ones: ",sess.run(ones))
    sess.close()
    return ones

