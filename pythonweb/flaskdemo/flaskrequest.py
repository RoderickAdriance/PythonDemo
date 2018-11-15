from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    username=request.cookies.get('username')
    request.cookies.setdefault('username','Salar')
    print(username)
    return render_template('hello.html',name=username)

@app.route('/test/')
def test():
    print('test')

if __name__ == '__main__':
    app.run(debug=True)
