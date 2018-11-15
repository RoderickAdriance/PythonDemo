import pandas as pd
import numpy as np
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])

#选取DataFrame中的数据
print(df['A'])
# print(df.A) #和上面效果相同

print(df[0:3])#选择跨越多行或多列

print(df.loc['20130102']) #标签来选择数据 loc

print(df.loc['20130102',['A','B']])

print(df)
print(df.iloc[3,1])#位置进行选择 iloc
# 13

print(df.iloc[3:5,1:3])
"""
             B   C
2013-01-04  13  14
2013-01-05  17  18
"""

#混合选选取
print(df.ix[:3,['A','C']])
"""
            A   C
2013-01-01  0   2
2013-01-02  4   6
2013-01-03  8  10
"""

print(df.ix[:3,['A','C']])#混合选取
"""
            A   C
2013-01-01  0   2
2013-01-02  4   6
2013-01-03  8  10
"""

print(df[df.A>8]) #判定后选取
"""
             A   B   C   D
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""