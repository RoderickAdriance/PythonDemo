import pandas as pd
import numpy as np


def t1():
    #创建Series
    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])  # Series，它是单一列

    population = pd.Series([827483, 947475, 123214])

    # 构建DataFrame ,  可以理解为一个表格
    data = pd.DataFrame({'City name': city_names, 'Population': population})  # 参数放map
    print(data)




def t2():
    #加载加利福尼亚的住房数据
    california_housing_dataframe=pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
    #保存数据
    california_housing_dataframe.to_csv("california_housing_dataframe.csv",index=False)

def t3():
    california_housing_dataframe=pd.read_csv("california_housing_dataframe.csv")
    print(california_housing_dataframe.describe())

def t4():
    # 创建Series
    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])  # Series，它是单一列

    population = pd.Series([827483, 947475, 123214])

    cities=pd.DataFrame({ 'City name': city_names, 'Population': population })

    print(type(cities['City name']))

    population=population/1000
    print(population)

def t5():
    #向现有 DataFrame 添加了两个 Series：
    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])  # Series，它是单一列

    population = pd.Series([827483, 947475, 123214,878])

    cities = pd.DataFrame({'City name': city_names, 'Population': population})

    cities['Area square miles']=pd.Series([46.87, 176.53, 97.92])

    cities['Population density']=cities['Population']/ cities['Area square miles']

    #城市以圣人命名。
    #城市面积大于 50 平方英里。
    cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(
        lambda name: name.startswith('San'))

    print(cities)

def t6():
    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])  # Series，它是单一列

    population = pd.Series([827483, 947475, 123214, 878])

    cities = pd.DataFrame({'City name': city_names, 'Population': population})

    #随机排列索引
    cities=cities.reindex(np.random.permutation(cities.index))

    print(cities)

t6()