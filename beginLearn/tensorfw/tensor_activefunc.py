import tensorflow as tf


#添加层  四个参数：输入值、输入的大小、输出的大小和激励函数
def add_layer(inputs, in_size, out_size, activation_func=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size])) #指定正太分布的数值中取出指定个数的值
    biases =tf.Variable(tf.zeros([1,out_size]))+0.1 #biases 推荐不为0,故加了0.1
    Wx_plus_b=tf.matmul(inputs,Weight)+biases

    #激励函数不为None 就把输出值再应用一次激励函数再输出
    if activation_func is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_func(Wx_plus_b)
    return outputs