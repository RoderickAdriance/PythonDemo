import threading

def th():
    activeT = threading.active_count()

    print(activeT)

    allT_info = threading.enumerate()

    print(allT_info)

def current_t():
    print(threading.current_thread())

if __name__ == '__main__':
    #target表示这个线程需要完成的任务
    thread = threading.Thread(target=current_t,)  # 定义线程
    thread.start()  # 让线程开始工作


