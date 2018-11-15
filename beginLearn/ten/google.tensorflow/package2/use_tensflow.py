import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.data import Dataset
from sklearn import metrics
import math
from matplotlib import pyplot as plt

# 以下代码会从 california_housing_dataframe 中提取 total_rooms 数据，
# 并使用 numeric_column 定义特征列，这样会将其数据指定为数值：

# 加载数据集
california_housing_dataframe = pd.read_csv("test.csv")
# 提取房间数
my_feature = california_housing_dataframe[["total_rooms"]]
# 获取tf张量中的 "total_rooms"数字列
feature_columns = [tf.feature_column.numeric_column("total_rooms")]  # total_rooms 数据的形状是一维数组（每个街区的房间总数列表）。
# 提取房子 价值
targets = california_housing_dataframe["median_house_value"]

# 优化器
# 1:梯度下降
# 2:梯度裁剪
optimizer1 = tf.train.GradientDescentOptimizer(0.0000001)
optimizer2 = tf.contrib.estimator.clip_gradients_by_norm(optimizer1, 5.0)

# 创建线性回归模型
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=optimizer2)


# 定义线性回归模型的输入函数
#TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。
#如果将默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。
#shuffle 设置为 True，则我们会对数据进行随机处理
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # features is a 'k,v' map
    features = {key: np.array(value) for key, value in dict(features).items()}
    #根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 batch_size 的多批数据，
    # 以按照指定周期数 (num_epochs) 进行重复。
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        #buffer_size 参数会指定 shuffle 将从中随机抽样的数据集的大小。
        ds = ds.shuffle(buffer_size=10000)

    #最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。
    #消耗迭代器中的值,get_next一次少一个
    feature, label = ds.make_one_shot_iterator().get_next()
    return feature, label


#将 my_input_fn 封装在 lambda 中，以便可以将 my_feature 和 target 作为参数传入，首先，我们会训练 100 步。
_ = linear_regressor.train(input_fn=lambda: my_input_fn(my_feature, targets), steps=100)

# ````````````````````评估模型Start```````````````
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
prediction = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in prediction])

# 均方误差, 预测值和目标值
mean_squared_error = metrics.mean_squared_error(predictions, targets)
#均方根误差
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)



#比较一下 RMSE 与目标最大值和最小值的差值：
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
# ````````````````````评估模型End```````````````


# calibration_data = pd.DataFrame()
# #预测值
# calibration_data["predictions"] = pd.Series(predictions)
# #真实值
# calibration_data["targets"] = pd.Series(targets)
# print(calibration_data.describe())


#
# # -----------画出图形-------------
sample = california_housing_dataframe.sample(n=500)
# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')


# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
# ------------图形绘画完毕--------------
