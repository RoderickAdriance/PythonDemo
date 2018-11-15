import json

#Json里面必须使用"" 双引号,单引号报错
# str = '''
# [{
#     "name": "Bob",
#     "gender": "male",
#     "birthday": "1992-10-18"
# }, {
#     "name": "Selina",
#     "gender": "female",
#     "birthday": "1995-10-18"
# }]
# '''
# print(type(str))
# data=json.loads(str)
# print(type(data))
# print(data)
# print(data[0]['name'])
# #Get方法可以传入键名,没有取到不会报错,而会返回None。get()方法可以传入第二个参数,就是默认值
# print(data[0].get('aa'))
# print(data[0].get('aa',50))


str =[{
    "name": "Bob",
    "gender": "male",
    "birthday": "1992-10-18"
}, {
    "name": "Selina",
    "gender": "female",
    "birthday": "1995-10-18"
},
    {
        "name": "中文",
        "gender": "男",
        "birthday": "1995-10-18"
    }
]

#dumps()方法，我们可以将JSON对象转为字符串
# with open('data.json','w') as file:
#     file.write(json.dumps(str))

#中文字符指定参数ensure_ascii为False
with open('data.json','w',encoding='utf-8') as file:
    file.write(json.dumps(str,ensure_ascii=False))

