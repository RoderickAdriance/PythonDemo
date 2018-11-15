import tensorflow as tf

#变量与占位符不同的一点是，变量在使用之前需要做初始化。

sess = tf.Session
d60 = tf.Variable(1, dtype=tf.float32, name='number1')
d61 = tf.tan(d60)
init_op = tf.global_variables_initializer()
sess.run(init_op)
sess.run(d61)

