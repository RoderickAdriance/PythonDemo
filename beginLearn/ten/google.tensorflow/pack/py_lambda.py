# python lambda 表达式

func = lambda x: x + 1  # 快速定义简单方法
func2 = lambda x, y: x + y  # 快速定义简单方法

# print(func(1))
# print(func2(2,3))


foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]

l = list(map(lambda x: x * 2, foo))#感觉和scala相同
print(l)
