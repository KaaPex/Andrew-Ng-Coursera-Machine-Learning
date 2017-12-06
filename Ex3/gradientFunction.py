import numpy as np
from sigmoid import sigmoid
from costFunction import costFunction


def gradientFunction(theta, X, y):
    """
    @brief      Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta

    @param      theta  The theta
    @param      X      features
    @param      y      label

    @return     gradient
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    grad = np.zeros(theta.size)

    h = sigmoid(X.dot(theta.T))
    grad = X.T.dot(h-y)/m

    return grad.T
