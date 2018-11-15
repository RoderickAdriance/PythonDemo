#object表示继承哪个类
class Student(object):
    def __init__(self,name,score):
        # ‘__’私有变量
        self.__name=name
        self.__score=score

    def get_name(self):
        return self.__name

    def get_score(self):
        return self.__score

    def print_score(self):
        print('%s: %s' % (self.__name,self.__score))

student = Student('a',5)
student.get_score()
student.print_score()
