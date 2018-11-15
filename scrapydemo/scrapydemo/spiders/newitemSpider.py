import scrapy

class newitemSprder(scrapy.Spider):
    name="newitemSprder"

    start_urls=['http://lab.scrapyd.cn']

    def parse(self, response):
        mingyan = response.css('div.quote')# 提取首页所有名言，保存至变量mingyan
        for v in mingyan:
            text = v.css('.text::text').extract_first()  # 提取名言
            autor = v.css('.author::text').extract_first()  # 提取作者
            tags = v.css('.tags .tag::text').extract()  # 提取标签
            tags = ','.join(tags)  # 数组转换为字符串
            fileName = '%s-语录.txt' % autor  # 定义文件名,如：木心-语录.txt
            with open(fileName, "a+") as f:  # 不同人的名言保存在不同的txt文档，“a+”以追加的形式
                f.write(text)
                f.write('\n')  # ‘\n’ 表示换行
                f.write('标签：' + tags)
                f.write('\n-------\n')
                f.close()

        next_page = response.css('li.next a::attr(href)').extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)

