import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from exercise.c10.tensortest.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

#Loss = (a^i - y^i)^2

y_hat = tf.constant(36,name='y_hat')
y=tf.constant(39,name='y')

loss=tf.Variable((y_hat-y)**2,name='loss')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(loss))
