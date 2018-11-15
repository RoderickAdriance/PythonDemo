import requests

r = requests.get('http://www.jianshu.com')
#成功就打印信息,失败直接调用 exit方法退出
exit() if not r.status_code == requests.codes.ok else print('Request Successfully')