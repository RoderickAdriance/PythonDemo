from lxml import etree
import requests
#//li[@data-index='4']//span[@class='title-link']/a[@target='_blank']
url="https://www.baidu.com/"

headers={
    "User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Mobile Safari/537.36",
    "Upgrade-Insecure-Requests":"1"
}

response = requests.get(url, headers=headers)

html_str = response.content.decode()

html = etree.HTML(html_str)

list = html.xpath("//span[@class='index-banner-text']/text()")

print(list)
