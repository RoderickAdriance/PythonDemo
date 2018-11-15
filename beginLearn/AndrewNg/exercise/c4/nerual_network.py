import matplotlib.pyplot as plt
from exercise.c4.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from exercise.c4.help_func import *
from exercise.c4.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)
X, Y = load_planar_dataset()
print("X.shape",X.shape)
print("Y.shape",Y.shape)

def show_scatter():
    plt.scatter(X[0], X[1], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.show()

def show_datashape():
    shape_X = X.shape
    shape_Y = Y.shape
    m=shape_X[1]
    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('I have m = %d training examples!' % (m))

def use_logisticRegressionCV():
    clf=sklearn.linear_model.LogisticRegressionCV();
    clf.fit(X.T,Y.T.ravel())
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")

def test_layersize():
    X_assess, Y_assess = layer_sizes_test_case()
    print("X_assess:", X_assess)
    print("Y_assess", Y_assess)
    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))

def test_init_parameters():
    n_x, n_h, n_y = initialize_parameters_test_case()

    parameters = initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

def test_forward_propagation():
    X_assess, parameters = forward_propagation_test_case()

    A2, cache = forward_propagation(X_assess, parameters)

    # Note: we use the mean here just to make sure that your output matches ours.
    print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

def test_compute_cost():
    A2, Y_assess, parameters = compute_cost_test_case()

    print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def test_backword_propagation():
    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
    grads = backward_propagation(parameters, cache, X_assess, Y_assess)
    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("db2 = " + str(grads["db2"]))

def test_update_parameters():
    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

def run_nn_model():
    X_assess, Y_assess = nn_model_test_case()

    parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

def test_predict():
    parameters, X_assess = predict_test_case()

    predictions = predict(parameters, X_assess)
    print(predictions)
    print("predictions mean = " + str(np.mean(predictions)))

def run_predict():
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    predictions = predict(parameters, X)
    print('Accuracy: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

def plt_figure():
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

def other_data():
    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}


    dataset = "noisy_circles"


    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "noisy_moons":
        Y = Y % 2

    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.show()
