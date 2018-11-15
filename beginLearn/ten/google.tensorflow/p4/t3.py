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

california_housing_dataframe = pd.read_csv("california_housing.csv")
california_housing_dataframe=california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

