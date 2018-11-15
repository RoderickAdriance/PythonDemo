import requests
from requests.auth import HTTPBasicAuth

r = requests.get('http://localhost:5000', auth=HTTPBasicAuth('username', 'password'))
print(r.status_code)


r = requests.get('http://localhost:5000', auth=('username', 'password'))
print(r.status_code)