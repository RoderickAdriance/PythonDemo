import requests

headers={
    "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1"}

def getParseUrl(url):
    response = requests.get(url, headers=headers)
    return response.content.decode()

def postParseUrl(url,data):
   session = requests.session()
   response = session.post(url, data, headers)
   return response.content.decode()

