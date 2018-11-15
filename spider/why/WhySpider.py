from why import UrlParse
import json
class WhySpider:
    def __init__(self):
        self.url = "http://m.wenhuayun.cn/wechatActivity/wcTopActivityList.do"

    def get_content_list(self,html_str):
        dict_data = json.loads(html_str)
        content_list = dict_data["data"]
        return content_list

    def save_content_list(self,data):
        with open("why.json","w",encoding="utf-8") as f:
            for content in data:
                f.write(json.dumps(content,ensure_ascii=False))
                f.write("\n")

    def run(self):
        data = {"tagId":"bfb37ab6d52f492080469d0919081b2b","Lon":"121.48","Lat":"31.22","pageIndex":0,"pageNum":20}
        html_str = UrlParse.postParseUrl(self.url, data)
        content_list = self.get_content_list(html_str)
        print(content_list)
        self.save_content_list(content_list)


if __name__ == '__main__':
    spider=WhySpider()
    spider.run()