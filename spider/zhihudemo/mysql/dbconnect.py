import pymysql

db = pymysql.connect(host='localhost',user='root',password='mysql123456789',db='python_database')
cursor = db.cursor()
# cursor.execute('select version()')
# data = cursor.fetchone()
# print(data)
cursor = db.cursor()

sql='insert into students(id,name,age,sex) values(%s,%s,%s)'
try:
    cursor.execute(sql,(1,'小明',30))
    db.commit()
except:
    db.rollback()
db.close()

