import requests
import json

# url="https://m.douban.com/rexxar/api/v2/subject_collection/tv_american/items?os=android&callback=jsonp3&start=0&count=8&loc_id=108288&_=1525747036608"
url="https://m.douban.com/rexxar/api/v2/subject_collection/tv_american/items?os=android&start=0&count=8&loc_id=108288&_=1525747036608"

headers={
    "User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Mobile Safari/537.36",
    "Referer":"https://m.douban.com/tv/"
}

response = requests.get(url, headers=headers)
content = response.content.decode()
json_content = json.loads(content)
print(json_content)
#ensure_ascii不实用ascii保存,直接保存中文
jj = json.dumps(json_content,ensure_ascii=False,indent=2)
with open("douban.txt","w",encoding="utf-8") as f:
    f.write(jj)

