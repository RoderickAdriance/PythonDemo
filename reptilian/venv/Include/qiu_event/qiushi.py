from lxml import etree
import requests
import json

class QiuShiSpider:

    def __init__(self):
        self.url_temp = "https://www.qiushibaike.com/8hr/page/{}/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36"
        }

    def get_url_list(self):#1.根据url的规律,构造url list
        url=[self.url_temp.format(i) for i in range(1,14)]
        return url

    def parse_url(self,url): #2.发送请求,获取响应
        print("parseing :",url)
        response = requests.get(url, headers=self.headers)
        return response.content.decode()

    def get_content_list(self,html_str): #3.提取数据
        html = etree.HTML(html_str)
        #1.分组
        div_list = html.xpath("//div[@id='content-left']/div")
        content_list=[]
        for div in div_list:
            item={}
            # 当前节点下的所有h2标签 文本
            item["author_name"]=div.xpath(".//h2/text()")[0].strip() if len(div.xpath(".//h2/text()")) >0 else None
            item["content"]=div.xpath(".//div[@class='content']/span/text()")
            item["content"]=[i.strip() for i in item["content"]]
            item["laugh"]=div.xpath(".//span[@class='stats-vote']/i/text()")[0] if len(div.xpath(".//span[@class='stats-vote']/i/text()")[0]) >0 else 0
            item["stats-comments"]=div.xpath(".//span[@class='stats-comments']//i/text()")
            item["stats-comments"]= item["stats-comments"][0] if len(item["stats-comments"]) >0 else None
            item["img"]= "https:"+div.xpath(".//div[@class='thumb']//img/@src")[0] if len(div.xpath(".//div[@class='thumb']//img/@src")) >0 else None
            #if 条件成立,就把if 前面的复制给 变量 ,不成立取后面的
            # item["img"]= "https:"+item["img"][0] if len(item["img"][0]) >0 else None
            content_list.append(item)
        return content_list

    def save_content_list(self,content_list):#保存
        with open("qiubai.txt","a",encoding="utf-8") as f:
            for content in content_list:
                f.write(json.dumps(content,ensure_ascii=False))
                f.write("\n")
        print("保存成功")

    def run(self):
        #1.根据url的规律,构造url list
        url_list=self.get_url_list()
        #2.发送请求,获取响应
        for url in url_list:
            html_str = self.parse_url(url)
        #3.提取数据
            content_list=self.get_content_list(html_str)
        #4.保存
        self.save_content_list(content_list)

if __name__ == '__main__':
    qiubai=QiuShiSpider()
    qiubai.run()