import threading


def threadmsg():
    print(threading.active_count())
    print(threading.enumerate())
    print(threading.current_thread())

def thTest():
    #添加线程，threading.Thread()接收参数target代表这个线程要完成的任务，需自行定义
    thread = threading.Thread(target=threadmsg, )  # 定义线程
    thread.start()  # 让线程开始工作


if __name__ == '__main__':
    thTest()
