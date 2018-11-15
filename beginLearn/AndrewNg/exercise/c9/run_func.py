import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from exercise.c9.opt_utils import *
from exercise.c9.testCases import *

def update_parameters_with_gd(parameters, grads, learning_rate):
    # 有W和b所以除2,获取层数
    L = len(parameters) // 2

    # 循环每一层,第0层是输入特征,故从第一层开始
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def test_update_parameters_with_gd():
    parameters, grads, learning_rate = update_parameters_with_gd_test_case()

    parameters = update_parameters_with_gd(parameters, grads, learning_rate)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
     """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    # 进行随机排列
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # 共分成几批,  向下取整,最后一批另作处理
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        # 取特征所有行,例 shuffled_X[:,1000*0,1000*1],shuffled_X[:,1000*1,1000*2]
        mini_batch_X = shuffled_X[:, mini_batch_size * k:mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k:mini_batch_size * (k + 1)]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 最后一批,没有到达mini_batch_size的
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def test_random_mini_batches():
    X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

    print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))


def initialize_velocity(parameters):
    """
       Initializes the velocity as a python dictionary with:
                   - keys: "dW1", "db1", ..., "dWL", "dbL"
                   - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
       Arguments:
       parameters -- python dictionary containing your parameters.
                       parameters['W' + str(l)] = Wl
                       parameters['b' + str(l)] = bl

       Returns:
       v -- python dictionary containing the current velocity.
                       v['dW' + str(l)] = velocity of dWl
                       v['db' + str(l)] = velocity of dbl
       """
    L = len(parameters) // 2
    v = {}
    # 初始化导数 全0
    for l in range(L):
        l = str(l + 1)
        v["dW" + l] = np.zeros(parameters["W" + l].shape)
        v["db" + l] = np.zeros(parameters["b" + l].shape)
    return v


def test_initialize_velocity():
    parameters = initialize_velocity_test_case()

    v = initialize_velocity(parameters)
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    # vdW[l]=βvdW[l]+(1−β)dW[l]
    # W[l]=W[l]−αvdW[l]
    # vdb[l]=βvdb[l]+(1−β)db[l]
    # b[l]=b[l]−αvdb[l]
    L = len(parameters) // 2

    for l in range(L):
        l = str(l + 1)
        v["dW" + l] = beta * v["dW" + l] + (1 - beta) * grads["dW" + l]
        v["db" + l] = beta * v["db" + l] + (1 - beta) * grads["db" + l]

        parameters["W" + l] = parameters["W" + l] - learning_rate * v["dW" + l]
        parameters["b" + l] = parameters["b" + l] - learning_rate * v["db" + l]

    return parameters, v


def test_update_parameters_with_momentum():
    parameters, grads, v = update_parameters_with_momentum_test_case()

    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        l = str(l + 1)
        v["dW" + l] = np.zeros(parameters["W" + l].shape)
        v["db" + l] = np.zeros(parameters["b" + l].shape)

        s["dW" + l] = np.zeros(parameters["W" + l].shape)
        s["db" + l] = np.zeros(parameters["b" + l].shape)

    return v, s


def test_initialize_adam():
    parameters = initialize_adam_test_case()

    v, s = initialize_adam(parameters)
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
        Update parameters using Adam

        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates
        beta2 -- Exponential decay hyperparameter for the second moment estimates
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing your updated parameters
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        l = str(l + 1)
        v["dW" + l] = beta1 * v["dW" + l] + (1 - beta1) * grads["dW" + l]
        v["db" + l] = beta1 * v["db" + l] + (1 - beta1) * grads["db" + l]

        s["dW" + l] = beta2 * s["dW" + l] + (1 - beta2) * grads["dW" + l]**2
        s["db" + l] = beta2 * s["db" + l] + (1 - beta2) * grads["db" + l]**2

        v_corrected["dW" + l] = v["dW" + l] / (1 - beta1 ** t) #t为训练批次
        v_corrected["db" + l] = v["db" + l] / (1 - beta1 ** t)

        s_corrected["dW" + l] = s["dW" + l] / (1 - beta2 ** t)
        s_corrected["db" + l] = s["db" + l] / (1 - beta2 ** t)

        parameters["W" + l] = parameters["W" + l] - learning_rate * v_corrected["dW" + l] / (
                np.sqrt(s_corrected["dW" + l]) + epsilon)

        parameters["b" + l] = parameters["b" + l] - learning_rate * v_corrected["db" + l] / (
                    np.sqrt(s_corrected["db" + l]) + epsilon)

    return parameters,v,s

def test_update_parameters_with_adam():
    parameters, grads, v, s = update_parameters_with_adam_test_case()
    parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0005, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    L=len(layers_dims)
    costs=[]
    t=0
    seed=10

    parameters=initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        seed=seed+1
        #数据分批
        minibatches =random_mini_batches(X,Y,mini_batch_size,seed)
        #对每一批数据做处理
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(a3, minibatch_Y)

            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


train_X, train_Y = load_dataset()

def test_model():
    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]

    parameters = model(train_X, train_Y, layers_dims, optimizer="adam")#gd  momentum  adam

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

test_model()