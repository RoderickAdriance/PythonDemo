import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50) #范围是(-3,3);个数是50
y1 = 2*x + 1
y2 = x**2

plt.figure()#定义一个图像窗口
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

new_ticks = np.linspace(-3, 3, 5) #范围是(-1,2);个数是5
print(new_ticks)
plt.xticks(new_ticks) #设置x轴刻度：范围是(-1,2);个数是5.
#设置y轴刻度以及名称,对应刻度的名称为[‘really bad’,’bad’,’normal’,’good’, ‘really good’]
plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
plt.show()