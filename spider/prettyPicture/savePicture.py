import os
from hashlib import md5
import requests
def save_image(item):
    title = 'picture/'+item.get('title').split(",")[0].replace("|","")
    if not os.path.exists(title):
        os.mkdir(title)
    try:
        url=item.get('image')
        if not url==None:
            response = requests.get(url)
            if response.status_code==200:
                #用于字符串格式化
                file_path='{0}/{1}.{2}'.format(title,md5(response.content).hexdigest(),'jpg')
                if not os.path.exists(file_path):
                    with open(file_path,'wb') as f:
                        f.write(response.content)

    except requests.ConnectionError:
        print('Failed to Save Image')

