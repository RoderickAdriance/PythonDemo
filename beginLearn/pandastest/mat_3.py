import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame(
    np.random.randn(500,4),
    index=np.arange(500),
    columns=list("ABCD")
    )
data.cumsum()

ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')

# 将之下这个 data 画在上一个 ax 上面
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()