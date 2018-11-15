import json
import os
from _md5 import md5
from multiprocessing.pool import Pool
from urllib.parse import urlencode,quote
import requests
from lxml import etree

from doubanJapan.agent import get_ip_list, get_random_ip, getIp


def getpage(start,title,ip):
    headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    }
    params = {
       'start':start,
        'type':'T'
    }
    title=title
    url="https://book.douban.com/tag/{title}?".format(title=quote(title))
    url=url+urlencode(params)
    response = requests.get(url, headers=headers,proxies=ip)
    if response.status_code==200:
        return response.content.decode()

def parseHtml(htmlStr):
    content_list=[]
    list = etree.HTML(htmlStr).xpath("//div/ul[@class='subject-list']/li")
    for li in list:
        item={}
        item["picture"]=li.xpath("./div/a/img/@src")
        item["title"]=li.xpath("./div/h2/a/@title")
        item["author"]=li.xpath("./div/div[@class='pub']/text()")[0].replace("\n","").replace(" ","")
        item["fraction"]=li.xpath(".//div[@class='star clearfix']/span[@class='rating_nums']/text()")
        item["evaluateCount"]=li.xpath(".//div[@class='star clearfix']/span[@class='pl']/text()")
        item["introduce"]=li.xpath("./div/p/text()")
        content_list.append(item)
        print(item)
    return  content_list

def savePicture(content_list):
    for item in content_list:
        purl=item["picture"][0]
        if purl:
            response = requests.get(purl)
            file_path = 'bookPicture/{0}.{1}'.format(md5(response.content).hexdigest(),'jpg')
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(response.content)

def saveBookInfo(content_list,title):
    if content_list:
        with open("{title}.txt".format(title=title),"a",encoding="utf-8") as f:
            f.write(json.dumps(content_list,ensure_ascii=False))
            f.write("\n")

def run(start):
    list=['日本文学','小说','随笔','散文','诗歌','童话','名著']
    for title in list:
        ip=getIp()
        htmlStr = getpage(start,title,ip)
        books = parseHtml(htmlStr)
        saveBookInfo(books,title)
        savePicture(books)

if __name__ == '__main__':
    # pool = Pool()
    # start=0
    # end=60
    # list=[i*20 for i in range(start,end)]
    # pool.map(run,list)
    # pool.join()
    # pool.close()
    list = [i * 20 for i in range(0, 60)]
    for start in list:
        run(start)
