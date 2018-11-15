import numpy as np
import tensorflow as tf

#1行2列
matrix1 = tf.constant([[3,3]])
#2行1列
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    print(sess.run(product))