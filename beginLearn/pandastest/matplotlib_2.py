import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame(
    np.random.randn(1000,5),
    index=np.arange(1000),
    columns=list("ABCDE")
    )
data.cumsum()
data.plot()
plt.show()
