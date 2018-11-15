import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from exercise.c6.dnn_app_utils_v2 import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


def show_image(index):
    plt.imshow(train_x_orig[index])
    print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
    plt.show()


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations

        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
    """
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)

        # dL/da
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # 反向传播
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
       Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

       Arguments:
       X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
       Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
       layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
       learning_rate -- learning rate of the gradient descent update rule
       num_iterations -- number of iterations of the optimization loop
       print_cost -- if True, it prints the cost every 100 steps

       Returns:
       parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters=update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters