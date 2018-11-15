import requests
from urllib.parse import urlencode
import json

def get_page(offset):
    params={
    'offset':offset,
    'format':'json',
    'keyword':'街拍',
    'autoload':'true',
    'count':20,
    'cur_tab':1
    }
    headers={
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    }

    #urlencode()方法构造请求的GET参数
    url='https://www.toutiao.com/search_content/?'+urlencode(params)
    try:
        response = requests.get(url,headers=headers)
        if response.status_code==200:
            return response.json()
    except requests.ConnectionError:
            return None

#提取图片
def get_images(json):
    if json.get('data'):
        for i in json.get('data'):
            title = i.get('title')
            images = i.get('image_list')
            #构造dict
            if not images==None:
                for image in images:
                    yield {
                        'image':'http:'+image.get('url'),
                        'title':title
                    }
