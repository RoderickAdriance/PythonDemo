import requests
from lxml import etree

url = 'https://www.zhihu.com/explore'
headers={
    "User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Mobile Safari/537.36",
    "Upgrade-Insecure-Requests":"1"
}

response = requests.get(url, headers=headers)
html_str = response.content.decode()
html = etree.HTML(html_str)
title = html.xpath("//h2/a[@data-za-element-name='Title']/text()")
print(title)
for item in title:
    with open('title.txt','a',encoding='utf-8') as f:
        f.write(item)
