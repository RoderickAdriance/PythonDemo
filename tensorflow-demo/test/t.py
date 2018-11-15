import tensorflow as tf
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32) #输入值 x
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)#输入值的预期输出值 y_true
#简单的线性模型，其输出值只有 1 个
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)
#损失函数
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
#优化函数, 梯度下降优化:根据损失相对于变量的导数大小来修改各个变量
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
    #执行优化函数 ,    输出 当前优化后的损失值
    _,loss_value = sess.run((train, loss))
print(loss_value)

print(sess.run(y_pred))
print(sess.run(y_pred))