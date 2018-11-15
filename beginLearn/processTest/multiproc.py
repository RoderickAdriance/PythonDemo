import multiprocessing as mp


def job(x):
    return x*x

def multicore():
    # 进程池
    pool = mp.Pool()
    #map()中需要放入函数和需要迭代运算的值，然后它会自动分配给CPU核，返回结果
    res = pool.map(job, range(10))
    print(res)

if __name__ == '__main__':
    multicore()