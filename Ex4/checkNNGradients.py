import numpy as np
from net import Net


def debugInitializeWeights(fan_out, fan_in):
    """initializes the weights of a layer with fan_in incoming connections
    and fan_out outgoing connections using a fix set of values
    Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    the first row of W handles the "bias" terms
    """

    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.reshape(np.sin(range(1, W.size+1)), W.T.shape).T / 10.0
    return W


def checkNNGradients(Lambda=0):

    """Creates a small neural network to check the
    backpropagation gradients, it will output the analytical gradients
    produced by your backprop code and the numerical gradients (computed
    using computeNumericalGradient). These two gradient computations should
    result in very similar values.
    """

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # print(Theta1.shape, Theta2.shape)

    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = np.mod(range(1, m+1), num_labels)
    # print(X.shape, y.shape)

    # Short hand for cost function
    net = Net(input_layer_size, hidden_layer_size, num_labels, Theta1, Theta2)
    numgrad = net.computeNumericalGradient(X, y, Lambda)
    _, grad = net.costFunction(X, y, Lambda)

    for key, _ in grad.items():
        # Visually examine the two gradient computations.  The two columns
        # you get should be very similar.
        print(np.column_stack((numgrad[key].T.ravel(), grad[key].T.ravel())))

        print('The above two columns you get should be very similar.\n' \
              '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

        # Evaluate the norm of the difference between two solutions.
        # If you have a correct implementation, and assuming
        # you used EPSILON = 0.0001
        # in computeNumericalGradient.m, then diff below should be less than 1e-9
        diff = np.linalg.norm(numgrad[key].T.ravel()-grad[key].T.ravel())
        diff /= np.linalg.norm(numgrad[key].T.ravel()+grad[key].T.ravel())

        print('If your backpropagation implementation is correct, then\n ' \
              'the relative difference will be small (less than 1e-9). \n' \
              '\nRelative Difference: %g\n' % diff)
