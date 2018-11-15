from __future__ import print_function

import math
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("data.csv")

# 数据随机化处理
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0

# 定义特征并配置特征列
my_feature = california_housing_dataframe[["total_rooms"]]

# total_rooms 数据的形状是一维数组（每个街区的房间总数列表）。
# 这是 numeric_column 的默认形状，因此我们不必将其作为参数传递。
feature_columns = [tf.feature_column.numeric_column("total_rooms")]#_NumericColumn(key='total_rooms', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)

targets = california_housing_dataframe["median_house_value"]

# 梯度优化梯度裁剪
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 线性回归模型
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)


# 定义定义输入函数
# 输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。
# 它告诉 TensorFlow 如何对数据进行预处理，
# 以及在模型训练期间如何批处理、随机处理和重复数据。
# shuffle 设置为 True，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。
# batch_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # 将 Pandas 特征数据转换成 NumPy 数组字典。
    features = {key: np.array(value) for key, value in dict(features).items()}
    # TensorFlow Dataset API 根据我们的数据构建 Dataset 对象
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 将 my_input_fn 封装在 lambda 中,以便可以将 my_feature 和 target 作为参数传入
fn = lambda: my_input_fn(my_feature, targets)
_ = linear_regressor.train(input_fn=fn, steps=100)

# 评估模型
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

predictions = linear_regressor.predict(input_fn=prediction_input_fn)

predictions = np.array([item["predictions"][0] for item in predictions])
# 均方误差
mean_squared_error = metrics.mean_squared_error(predictions, targets)
# 开根号
root_mean_squared_error = math.sqrt(mean_squared_error)

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value
print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data)
print(calibration_data.describe())

# 我们将获得均匀分布的随机数据样本300个，以便绘制可辨的散点图。
sample = california_housing_dataframe.sample(n=300)

# 根据模型的偏差项和特征权重绘制学到的线，并绘制散点图。该线会以红色显示。

x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# 获取第一个特征的权重
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

print("weight:",weight)
print("bias:",bias)
#进行预测
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

plt.plot([x_0,x_1],[y_0,y_1],c='r')

plt.xlabel("total_rooms")
plt.ylabel("median_house_value")

plt.scatter(sample["total_rooms"],sample["median_house_value"])

plt.show()
