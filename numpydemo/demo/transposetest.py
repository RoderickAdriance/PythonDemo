import numpy

def j():
    # Matrix.Transpose 矩阵转置   两行两列的矩阵
    mat = numpy.mat([[1, 2], [3, 4]])
    print(mat.T)


def a():
    print("Matrix Multiplication")
    a = numpy.mat([1, 2])
    b = numpy.mat([[10], [20]])
    print(a * b)
    print(a.T * b.T)

def inner():
    #          n
    #xy=⟨x,y⟩=∑     xiyi
    #          i=1
    x = numpy.array([1, 2])
    y = numpy.array([10, 20])
    print("Array inner:")
    print(numpy.inner(x, y))

    x = numpy.mat([[1, 2], [3, 4]])
    y = numpy.mat([10, 20])
    print("Matrix inner:")
    print(numpy.inner(x, y))

def l():
    x = numpy.array([1, 3])
    y = numpy.array([10, 20])
    print("Array outer:")
    print(numpy.outer(x, y))

    x = numpy.mat([[1, 2], [3, 4]])
    y = numpy.mat([10, 20])
    print("Matrix outer:")
    print(numpy.outer(x, y))


if __name__ == '__main__':
    l()
