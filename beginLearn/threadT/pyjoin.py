import threading
import time

def thread_job():
    print("T1 start")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish")

def T2_job():
    print("T2 start\n")
    print("T2 finish\n")

#不添加join 主线程会不等待子线程执行完毕就退出。
if __name__ == '__main__':
    thread_1 = threading.Thread(target=thread_job, name='T1')
    thread_2 = threading.Thread(target=T2_job, name='T2')
    thread_1.start()  # 开启T1
    #等待t1完成再执行下面的代码
    thread_1.join()

    thread_2.start()  # 开启T2
    print("all done\n")