from flask import Flask
#参数通常是主模块或是包的名字
#Flask用这个参数来决定程序的根目录
# 以便以后找到资源文件，比如网页中的图片，视频，音频等
app = Flask(__name__)# 创建一个Web应用

@app.route('/')# 定义路由(Views)，可以理解为定义页面的URL
def index():
    return "Hello World!"# 渲染页面

if __name__=="__main__":
    app.run(host='localhost', port=8081,debug=True)  # 运行，指定监听地址为 127.0.0.1:8080
