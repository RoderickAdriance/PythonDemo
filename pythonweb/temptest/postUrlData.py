from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/get/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')

        print("POST 方法 \nname: %s \nage: %s" % (name, age))

        return '已经获取数据'


if __name__ == '__main__':
    app.run(debug=True)