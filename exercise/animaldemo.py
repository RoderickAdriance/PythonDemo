class Animal(object):
    def run(self):
        print('Animal can run')

class Dog(Animal):
    def eat(self):
        print('Dog like eat shit')

    def __len__(self):
        return 100

def run_animal(Animal):
    Animal.run()

dog = Dog()
print(len(dog))