import pandas as pd
import numpy as np
s = pd.Series([1,3,6,np.nan,44,1])

print(s)
"""
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
dtype: float64
"""

dates = pd.date_range('20160101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])

print(df)
print(df['b'])

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))#创建一组没有给定行标签和列标签的数据 df1
print(df1)


df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
print(df2)
print(df2.dtypes)
print(df2.index)
print(df2.columns)
print(df2.describe())#数据的总结
print(df2.T)#翻转数据, transpose:
print(df2.sort_index(axis=1, ascending=False))#数据的 index 进行排序并输出
print(df2.sort_values(by='B')) #如果是对数据 值 排序输出