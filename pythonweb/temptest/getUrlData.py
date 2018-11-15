from flask import Flask, request

app = Flask(__name__)

@app.route('/get/')
def index():
    name = request.args.get('name')
    age = request.args.get('age')
    print("name:%s \n age:%s" % (name,age))
    return "OK"

if __name__ == '__main__':
    app.run(debug=True)
