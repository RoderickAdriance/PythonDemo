from flask import Flask, render_template

app = Flask(__name__)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    print(name)
    return render_template('hello.html',name=name)

#默认接受参数是String
@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s' % username
#指定接受的参数类型为int
@app.route('/post/<int:post_id>')
def show_post(post_id):
    print(type(post_id))
    return 'Post %d' % post_id



if __name__ == '__main__':
    app.logger.debug('A value for debugging')
    app.logger.warning('A warning occurred (%d apples)', 42)
    app.logger.error('An error occurred')
    app.run()

