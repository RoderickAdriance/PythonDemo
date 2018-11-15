import tensorflow as tf

sess = tf.Session()
# initial = tf.truncated_normal([6, 6], stddev=0.5)
#
# print(sess.run(initial))

constant = tf.constant(0.1, shape=[6,6])
print(sess.run(constant))