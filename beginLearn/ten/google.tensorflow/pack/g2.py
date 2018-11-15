import pandas as pd

#但是在大多数情况下，您需要将整个文件加载到 DataFrame 中。下面的示例加载了一个包含加利福尼亚州住房数据的文件。
# 请运行以下单元格以加载数据，并创建特征定义：
california_housing_dataframe=pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe.describe()

california_housing_dataframe.head()

#pandas 的另一个强大功能是绘制图表。例如，借助 DataFrame.hist，您可以快速了解一个列中值的分布：
california_housing_dataframe.hist('housing_median_age')