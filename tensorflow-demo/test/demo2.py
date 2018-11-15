import tensorflow as tf
#创建一个变量,初始化标量 0
state = tf.Variable(0, name="counter")

#创建一个op,作用是使state增加1
#创建一个常量
one = tf.constant(1)
#做一个加法操作
new_value = tf.add(state, one)
#分配变量给state
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

#启动图,运行op
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 打印state的初始值
    print(sess.run((state)))
    #运行op ,更新state 打印state
    for _ in range(5):
        sess.run(update)#这里run的时候 调用了update里面的new_value
        print(sess.run(state))#取得state当前值

