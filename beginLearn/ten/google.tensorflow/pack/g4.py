import numpy as np
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({'City name': city_names, 'Population': population})

cities = cities.reindex([2, 0, 1])
print(cities)
#如果没有第3 列,他的值就是NaN
cities=cities.reindex([3,0,1,2])

print(cities)
