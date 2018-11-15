from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
import csv

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
#加载数据集
california_housing_dataframe = pd.read_csv("test.csv")

#随机索引列
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0

describe = california_housing_dataframe.describe()
print(describe)
