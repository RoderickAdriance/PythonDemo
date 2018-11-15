from __future__ import print_function

from matplotlib import pyplot as plt

import pandas as pd


california_housing_dataframe = pd.read_csv("california_housing.csv")
plt.scatter(california_housing_dataframe["latitude"],california_housing_dataframe["median_house_value"])
plt.show()