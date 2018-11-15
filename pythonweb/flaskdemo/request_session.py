from flask import Flask, session, redirect, url_for, escape, request, jsonify

app= Flask(__name__)

@app.route('/')
def index():
    if 'username' in session:
        return 'logged in as %s' % escape(session['username'])
    return  'You are not logged in'

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        session['username']=request.form['username']
        return redirect(url_for('index'))
    return ''

@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('index'))
@app.route('/testjson/')
def jsontest():
    return  jsonify(msg='hello world!')

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

if __name__ == '__main__':
    app.run(debug=True)