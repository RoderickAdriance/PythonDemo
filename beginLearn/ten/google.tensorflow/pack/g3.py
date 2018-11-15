import pandas as pd
import numpy as np

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({'City name': city_names, 'Population': population})
# 增加新的列
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']

print(cities)

# Area square greater than 50
#城市名以 San 开头的
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50)  & cities['City name'].apply(lambda name: name.startswith('San'))

print(cities)
# cities.reindex([2, 0, 1])#重新排序
#输出索引值
print(city_names.index)

#调用numpy的函数进行随机排列
reindex = cities.reindex(np.random.permutation(city_names.index))
print(reindex)