from __future__ import print_function

import math
from sklearn import metrics
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
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


def my_input_fn(features, targets, batchsize=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batchsize).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, label = ds.make_one_shot_iterator().get_next()

    return features, label


features,label = my_input_fn(california_housing_dataframe[["total_rooms"]], california_housing_dataframe[["median_house_value"]],
                 batchsize=1)
