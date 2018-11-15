import pymongo
from bson.objectid import ObjectId
client = pymongo.MongoClient(host='hadoop', port=27017)
db = client['test']
collection = db['students']

def insertmongo():
    student1 = {
        'id': '20170101',
        'name': 'Jordan',
        'age': 20,
        'gender': 'male'
    }

    student2 = {
        'id': '20170202',
        'name': 'Mike',
        'age': 21,
        'gender': 'male'
    }

    result = collection.insert_many([student1, student2])
    print(result)

def querymongo():
    one = collection.find_one({'name': 'Mike'})
    print(one)
    result = collection.find_one({'_id': ObjectId('5be3b01b1a910103d8ed4875')})
    print(result)
    results = collection.find({'age': 20})
    for result in results:
        print(result)

    results = collection.find({'age': {'$gt': 20}})
    for result in results:
        print(result)

    count = collection.count()
    print(count)
    results = collection.find().sort('name', pymongo.ASCENDING).skip(2).limit(2)
    print([result['name'] for result in results])

if __name__ == '__main__':
    querymongo()

