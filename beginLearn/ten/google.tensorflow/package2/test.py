import pandas as pd
import tensorflow as tf
#以下代码会从 california_housing_dataframe 中提取 total_rooms 数据，
# 并使用 numeric_column 定义特征列，这样会将其数据指定为数值：

#加载数据集
california_housing_dataframe = pd.read_csv("test.csv")
#提取房间数
my_feature = california_housing_dataframe[["total_rooms"]]
#为total_rooms配置 数字类型的数据
feature_columns=[tf.feature_column.numeric_column("total_room")]#total_rooms 数据的形状是一维数组（每个街区的房间总数列表）。
print(feature_columns)

#提取房子 价值
targets = california_housing_dataframe["median_house_value"]

print(targets)
