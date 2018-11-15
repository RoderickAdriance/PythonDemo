
str='-'.join(['a','b','v'])
print(str)

data = {
    'id': '20120001',
    'name': 'Bob',
    'age': 21
}
keys = ', '.join(data.keys())
values = ', '.join(['%s'] * len(data))
print(values)