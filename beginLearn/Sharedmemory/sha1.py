import multiprocessing as mp

#使用Value数据存储在一个共享的内存表中
# 'd' 'i' 为数据类型
value1 = mp.Value('i', 0)
value2 = mp.Value('d', 3.14)
array = mp.Array('i', [1, 2, 3, 4])



