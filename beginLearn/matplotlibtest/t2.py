import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)#范围是(-1,1);个数是50. 仿真一维数据组(x ,y)表示曲线
y = 2*x + 1
y2 = x**2
plt.figure(num=3, figsize=(8, 5),) #编号为3；大小为(8, 5)
plt.plot(x, y,color='red', linewidth=1.0, linestyle='--')
plt.plot(x,y2)
plt.show()