import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    """
        COSTFUNCTION Compute cost and gradient for logistic regression
        computes the cost of using theta as the
        parameter for logistic regression and the gradient of the cost
        w.r.t. to the parameters.
    """

    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    h = sigmoid(X.dot(theta.T))
    J = -1 * np.sum(y*np.log(h)+(1-y)*np.log(1-h))/m

    return J
