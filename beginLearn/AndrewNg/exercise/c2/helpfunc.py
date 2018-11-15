import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    # print("After sigmoid:", s)
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    print("w:", w)
    print("b:", b)
    return w, b


def propagate(w, b, X, Y):
    # 正向传播
    m = X.shape[1]

    z=np.dot(w.T, X) + b

    A = sigmoid(z)

    cost = 1 / m * np.sum(-Y * np.log(A) - (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(X, (A - Y).T)

    db = 1 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    # m = X.shape[0]
    # y_hat = np.zeros((1, m))

    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    A = A > 0.5

    return A + 0


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # 初始化12288个特征权重
    w, b = np.zeros((X_train.shape[0], 1)), 0

    # parameters:  {"w": w, "b": b}  |  grads: {"dw": dw, "db": db}
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d
