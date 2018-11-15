import pandas as pd

california_housing_dataframe = pd.read_csv("data.csv")

print(california_housing_dataframe.describe())

hist = california_housing_dataframe.hist('housing_median_age')

