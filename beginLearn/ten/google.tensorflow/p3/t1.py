from __future__ import print_function
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

california_housing_dataframe = pd.read_csv("test.csv")

# 对数据进行随机化处理,打乱顺序, median_house_value 调整为以千为单位，
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

california_housing_dataframe["median_house_value"] /= 1000.0

# 从 california_housing_dataframe 中提取 total_rooms 数据
my_feature = california_housing_dataframe[["total_rooms"]]
# numeric_column 定义特征列，这样会将其数据指定为数值
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# 定义目标，也就是 median_house_value。从 california_housing_dataframe中提取它
targets = california_housing_dataframe["median_house_value"]

# 梯度优化
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 线性回归模型
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)


# ------定义输入函数---------
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features)}

    ds = Dataset.from_tensor_slices((features, targets))
    # batch_size 表示数据拆分成几份
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=100000)
    features, lables = ds.make_one_shot_iterator().get_next()
    return features,lables


