import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({'City name': city_names, 'Population': population})

print(type(cities['City name']))

print(cities['City name'][0])

print(cities[0:2])

apply = population.apply(lambda salary: salary > 862469)
print(apply)

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']



cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))

print(cities)

print(type(cities.index))

reindex = cities.reindex([0, 3, 4, 2])
print(reindex)
