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


def t1():
    california_housing_dataframe = pd.read_csv("california_housing_dataframe.csv")
    # 对数据做随机化处理
    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index)
    )

    california_housing_dataframe['median_house_value'] /= 1000.0

    print(california_housing_dataframe.describe())


def t2():
    california_housing_dataframe = pd.read_csv("california_housing_dataframe.csv")
    my_feature = california_housing_dataframe[["total_rooms"]]
    print(my_feature)
    # 使用 numeric_column 定义特征列，这样会将其数据指定为数值
    feature_columns = [tf.feature_column.numeric_column("total_rooms")]

    # Define the label
    targets = california_housing_dataframe["median_house_value"]# This is a Series
    targets = california_housing_dataframe[["median_house_value"]]

    #Configuration the optimizer
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer=tf.clip_by_norm(optimizer,5)

    #Create linear regression model
    linear_regressor=tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=optimizer
    )

def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    #将Pandas 特征数据转换成NumPy 数组字典
    features={key:np.array(value) for key,value in dict(features).items()}

    #使用TensorFlow Api 构建DataSet 将数据拆分batch_size 大小
    # 按照指定周期num_epochs重复
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels


