import requests
"""
专门请求url地址的方法
"""
headers={
    "User-Agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Mobile Safari/537.36",
    "Referer":"https://m.douban.com/tv/"
}


def parse_url(url):
    print("*")
    response = requests.get(url,headers=headers,timeout=5)
    return  response.content.decode()

if __name__ == '__main__':
    url="http://www.baidu.com"
    print(parse_url(url))