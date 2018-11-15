import json
from csdn import UrlParse
import time
class RecommendSpider:

    def __init__(self):
        self.url="https://www.csdn.net/api/articles?type=more&category=home&shown_offset={}"

    def get_content_list(self, html_str):  # 提取数据
        dict_data = json.loads(html_str)
        articles_ = dict_data["articles"]
        return articles_

    def save_content_list(self,content_list):
        with open("csdn.json","w",encoding="utf-8") as f:
            for content in content_list:
                if "desc" in content:
                    content["desc"]=str(content["desc"].strip).replace("\n","")
                f.write(json.dumps(content,ensure_ascii=False))
                f.write("\n")

    def run(self):
        # 1.根据url的规律,构造url list
        ticks = time.time()
        ticks=str(ticks).replace(".","")
        #1525852538778086
        url=self.url.format(ticks)
        # 2.发送请求,获取响应
        html_str = UrlParse.parseUrl(url)
        print(html_str)
        # 3.提取数据
        articles = self.get_content_list(html_str)
        # 4.保存
        self.save_content_list(articles)

if __name__ == '__main__':
    spider = RecommendSpider()
    spider.run()