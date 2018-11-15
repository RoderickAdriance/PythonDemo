import copy
a=[1,2,3]
b=a
c=copy.copy(a)
#输出地址
print(id(a))
print(id(b))
print(id(c))
print(a==b)
#虽然地址不同但数值类型比较数字,所以为true
print(a==c)