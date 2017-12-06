from costFunction import costFunction
import numpy as np


def costFunctionReg(theta, X, y, Lambda):
    """
    @brief      Compute cost and gradient for logistic regression with
                regularization

    @param      theta   The theta
    @param      X       features
    @param      y       target
    @param      Lambda  The lambda

    @return     the cost
    """
    J = 0
    m = y.size

    # skip x_0
    theta_ = theta[1:]

    J = costFunction(theta, X, y) + Lambda*np.sum(theta_**2) / (2*m)

    return J
