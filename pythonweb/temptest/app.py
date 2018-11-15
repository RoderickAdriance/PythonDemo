from flask import Flask,render_template,url_for,redirect
# Flask(__name__,template_folder=”存放模板文件夹名称”)
app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("index.html",name='Stronger')

@app.route('/user/')
def user():
    print(url_for('index', id=10, name='XeanYu', age=16))
    return 'PythonDemo'

#重定向
@app.route('/',methods=['GET','POST'])
def index():
    return redirect(url_for('user'))

if __name__ == '__main__':
    app.run(debug=True)

