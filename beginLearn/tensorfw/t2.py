import tensorflow as tf
import numpy as np

# create two matrixes
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)#matrix multiply

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
# [[12]]

# method 2    python打开用完后自动关闭
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)