import numpy as np


def computeCost(X, y, theta=np.zeros((2, 1))):
    """
       Сomputes the cost of using theta as the parameter for linear
       regression to fit the data points in X and y
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    return np.sum(np.square(X.dot(theta) - y))/(2*m)
# =========================================================================
