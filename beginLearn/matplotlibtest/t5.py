import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50) #范围是(-3,3);个数是50
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.xlim((-1, 2)) #x坐标轴范围：(-1, 2)
plt.ylim((-2, 3)) #y坐标轴范围：(-2, 3)

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks) #x轴刻度：范围是(-1,2) 个数是5
plt.yticks([-2, -1.8, -1, 1.22, 3],['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])


ax = plt.gca() #获取当前坐标轴信息
ax.spines['right'].set_color('none') #.spines设置边框：右侧边框
ax.spines['top'].set_color('none')
plt.show()

# #设置x坐标刻度数字或名称的位置：bottom.（所有位置：top，bottom，both，default，none）
ax.xaxis.set_ticks_position('bottom')
#x轴；使用.set_position设置边框位置：y=0的位置；（位置所有属性：outward，axes，data）
ax.spines['bottom'].set_position(('data', 0))


#设置y坐标刻度数字或名称的位置：left.（所有位置：left，right，both，default，none）
ax.yaxis.set_ticks_position('left')
#y轴；使用.set_position设置边框位置：x=0的位置；（位置所有属性：outward，axes，data
ax.spines['left'].set_position(('data',0))
plt.show()

