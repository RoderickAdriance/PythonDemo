import pandas as pd
import numpy as np

number_name=pd.Series(['one','two','three'])#Series，它是单一列。

number=pd.Series([1,2,3])
#可以将映射 string 列名称的 dict 传递到它们各自的 Series，从而创建DataFrame对象。
figure=pd.DataFrame({'number name': number_name,'number':number})

print(type(figure['number']))
print(figure['number name'][0])

np.log(figure['number']/1000)

figure['number']=pd.Series([5,6,7])

print(figure['number'])