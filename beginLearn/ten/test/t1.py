import tensorflow as tf

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

result = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print(sess.run(result))
