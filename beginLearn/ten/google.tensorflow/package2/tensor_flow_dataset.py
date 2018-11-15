import tensorflow as tf
import numpy as np
import pandas as pd

# uniform = tf.random_uniform([10])
# print(uniform)

#单个张量
# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
# print(dataset1.output_types)  # ==> "tf.float32"
# print(dataset1.output_shapes)  # ==> "(10,)"

#张量元组
# dataset2 = tf.data.Dataset.from_tensor_slices(
#    (tf.random_uniform([4]),tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
# print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
# print(dataset2.output_shapes)  # ==> "((), (100,))"

# 嵌套张量
# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
# print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
# print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

# dataset = tf.data.Dataset.from_tensor_slices(
#    {"a": tf.random_uniform([4]),"b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
# print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
# print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

# dataset = tf.data.Dataset.range(100)
# # iterator = dataset.make_one_shot_iterator()
# # next_element = iterator.get_next()
# # with tf.Session() as sess:
# #    for i in range(100):
# #      value = sess.run(next_element)
# #      assert i == value


# dataset = tf.data.Dataset.range(5)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# result = tf.add(next_element, next_element)
# with tf.Session() as sess:
#    sess.run(iterator.initializer)
#    print(sess.run(result))  # ==> "0"
#    print(sess.run(result))  # ==> "2"
#    print(sess.run(result))  # ==> "4"
#    print(sess.run(result))  # ==> "6"
#    print(sess.run(result))  # ==> "8"
#    try:
#      sess.run(result) #超过了iterator最大个数就会OutofRange
#    except tf.errors.OutOfRangeError:
#      print("End of dataset")  # ==> "End of dataset"


#如果数据集的每个元素都具有嵌套结构，
# 则 Iterator.get_next() 的返回值将是一个或多个 tf.Tensor 对象，这些对象具有相同的嵌套结构：
# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
# dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
#
# iterator = dataset3.make_initializable_iterator()
# with tf.Session() as sess:
#    sess.run(iterator.initializer)
#    next1, (next2, next3) = iterator.get_next()


# 加载数据集
california_housing_dataframe = pd.read_csv("test.csv")
# 提取房间数
features = california_housing_dataframe[["total_rooms"]]

labels = california_housing_dataframe[["median_house_value"]]


dataset = tf.data.Dataset.from_tensor_slices((features, labels))



#输入 features 行  labels  行
# print(features.shape[0] , labels.shape[0])
#
# features, labels  = dataset.make_one_shot_iterator().get_next()
# #输入 get_next() 方法取出一行
# print(features.shape[0] , labels.shape[0])
# with tf.Session() as sess:
#    #打印获取到的值
#    for i in range(10):
#       print(sess.run([features,labels]))

# d = dict(features).items()
# print(d)
# features = {key: np.array(value) for key, value in dict(features).items()}
# print(features)