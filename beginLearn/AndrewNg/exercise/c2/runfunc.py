import numpy as np
from exercise.c2.helpfunc import *
from exercise.c2 import lr_utils
from exercise.c2.handle_image import *


def run1():
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    grads, cost = propagate(w, b, X, Y)

    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("cost = " + str(cost))


def run2():
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=True)

    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))


def run3():
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=True)

    y_hat = predict(params["w"], params["b"], X)
    print(y_hat)

def run4(picture):
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig[0].shape[0]
    # 将 64x64x3 折叠为209,12288   (图片数,每张图片的纬度数)，然后转置
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    print("train_set_x:", train_set_x.shape)
    print("test_set_x:", test_set_x.shape)

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=False)

    w = d["w"]
    b = d["b"]

    pic_set=handle(picture)
    pic_set = pic_set.reshape(1, -1)
    pic_set = pic_set.T
    pic_set =pic_set/255
    print(pic_set.shape)

    predict1 = predict(w,b,pic_set)
    if predict1==1:
        print("猫🐱")
    else:
        print("不是猫")


run4("46ed19c65cb9160eaeea14efd3423501.jpg")
