from flask import Flask,session,request,render_template
import os

app = Flask(__name__)

#配置
app.config.update(
    DEBUG=True,
    SECRET_KEY=os.urandom(24)  # 使用os.urandom生成随机，进行加密
)
@app.route('/',methods=['GET','POST'])
def index():
    #判断是否是GET
    if request.method == 'GET':
        return render_template('blog.html',id=session.get('id'))# 用get,如果id不存在就不会返回异常，而是None

    # 如果不是GET,就是 POST
    session['id'] = request.form.get('id')
    return render_template('blog.html',id=session.get('id'))

@app.errorhandler(404)
def error_404(error):
    return '404 Not Found',404

if __name__ == '__main__':
    app.run()
