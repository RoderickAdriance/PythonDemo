import pymysql  # 导入 pymysql

# 打开数据库连接
db = pymysql.connect(host="localhost", user="root",
                     password="mysql123456789", db="python_database", port=3306)

# 使用cursor()方法获取操作游标
cur = db.cursor()

# 1.查询操作
# 编写sql 查询语句  user 对应我的表名
sql = "select * from employee"
try:
    cur.execute(sql)  # 执行sql语句

    results = cur.fetchall()  # 获取查询的所有记录
    print("id", "name", "password")
    # 遍历结果
    for row in results:
        name1 = row[0]
        name2 = row[1]
        age = row[2]
        print(name1, name2, age)
except Exception as e:
    raise e
finally:
    db.close()  # 关闭连接