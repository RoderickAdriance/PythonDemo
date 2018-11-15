import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# 初始化w和b
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b


# 向前传播向后传播
def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)

    # 对数损失   log(1)=0
    allcost = -Y * np.log(A) - (1 - Y) * np.log(1 - A)
    sumcost = np.sum(allcost)
    cost = 1 / m * sumcost

    # 可以理解为12228个像素，每个像素的平均误差
    dw = 1 / m * np.dot(X, (A - Y).T)

    db = 1 / m * np.sum(A - Y)

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
    m = X.shape[1]

    #(12288,1)
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T,X)+b)

    A = A >0.6
    return A+0

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b = np.zeros((X_train.shape[0],1)) ,0

    #计算后的参数梯度损失
    parameters, grads, costs= optimize(w,b,X_train,Y_train,num_iterations,learning_rate)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test=predict(w,b,X_test)

    #  预测减去真实,[1,0,,0,0,1,1,1,1,0] 取平均值
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

