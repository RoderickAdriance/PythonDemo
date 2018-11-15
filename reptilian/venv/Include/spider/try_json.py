import requests
import json

url="http://fanyi.baidu.com/v2transapi"
query_string={
    "from":"en",
    "to":"zh",
    "query":"你好"
}

headers={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36"}

response = requests.post(url, data=query_string, headers=headers)

html_str=response.content.decode()

dict_ret = json.loads(html_str)
print(dict_ret)
print(type(dict_ret))

ret=dict_ret["query"]
print(ret)
