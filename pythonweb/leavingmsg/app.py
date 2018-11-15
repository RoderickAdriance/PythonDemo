import time

from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

users=[]#这里存放所有的留言
@app.route('/index/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html',says=users)

    if request.method == 'POST':
        title = request.form.get('say_title')
        text = request.form.get('say')
        user = request.form.get('say_user')
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        users.append({
            "title":title,
            "user":user,
            "text":text,
            "date":date
        })
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
