from cmath import pi

from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    headers = request.headers
    host = headers.get('Host')
    addr = request.remote_addr
    agent = request.user_agent
    print(host,addr,agent)
    return 'Console'

if __name__ == '__main__':
    print(pi)
    # app.run(debug=True)
