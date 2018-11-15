import scrapy

class simpleUrl(scrapy.Spider):
    name="simpleUrl"
    start_urls = [  # 另外一种写法，无需定义start_requests方法
        'http://lab.scrapyd.cn/page/1/',
        'http://lab.scrapyd.cn/page/2/',
    ]

    def parse(self, response):
            page = response.url.split("/")[-2]
            filename = 'mingyan-%s.html' % page
            with open(filename, 'wb') as f:
                f.write(response.body)
            self.log('保存文件: %s' % filename)