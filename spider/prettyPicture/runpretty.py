from prettyPicture import pretty
from prettyPicture import savePicture
from multiprocessing.pool import Pool

def offset(offset):
    #发送一个请求获取相应
    json = pretty.get_page(offset)
    #提取页面中的图片
    for item in pretty.get_images(json):
        print(item)
        savePicture.save_image(item)


GROUP_START = 0
GROUP_END = 30
#使用多线程访问
if __name__ == '__main__':
    pool=Pool()
     # 列表推导式 [1*20,2*20,3*20......]
    groups=([x * 20 for x in range(GROUP_START,GROUP_END)])
    #map意义为列表中的每一个对象多线程调用方法offset
    pool.map(offset,groups)
    pool.close()
    pool.join()

