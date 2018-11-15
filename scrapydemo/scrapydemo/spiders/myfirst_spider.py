import scrapy

class firstspider(scrapy.Spider):
    name = "super_spider"
    def start_requests(self):
        urls=[
            'http://lab.scrapyd.cn/page/1/',
            'http://lab.scrapyd.cn/page/2/',
        ]
        for url in urls:
            yield scrapy.Request(url=url,callback=self.parse)

    def parse(self,response):
        page = response.url.split("/")[-2]  #根据上面的链接提取分页
        filename = 'mingyan-%s.html' % page #拼接文件名，如果是第一页，最终文件名便是：mingyan-1.html
        with open(filename, 'wb') as f:
            f.write(response.body)

        self.log('保存文件: %s' % filename)