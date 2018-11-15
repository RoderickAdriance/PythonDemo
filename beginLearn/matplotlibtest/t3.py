import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50) #范围是(-3,3);个数是50
y1 = 2*x + 1
y2 = x**2

plt.figure()#定义一个图像窗口
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

plt.xlim((-1, 2))   #设置x坐标轴范围：(-1, 2)
plt.ylim((-2, 3))   #设置y坐标轴范围：(-2, 3)
plt.xlabel('I am x')#设置x坐标轴名称：’I am x’
plt.ylabel('I am y')#y坐标轴名称：’I am y’
plt.show()

