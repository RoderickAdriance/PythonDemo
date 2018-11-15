import pandas as pd
import numpy as np

city_names=pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({'City name': city_names, 'Population': population})

california_housing_dataframe=pd.read_csv('data.csv')

california_housing_dataframe.hist('housing_median_age')

# population=population/1000

#log 实际上不是以10为底，而是以 e 为底
log_population = np.log(population)

apply = population.apply(lambda val: val > 500000)

cities['Area square miles']=pd.Series([46.87, 176.53, 97.92])
cities['Population density']=cities['Population']/cities['Area square miles']

reindex = cities.reindex([0, 5, 2, 8])
print(reindex)
