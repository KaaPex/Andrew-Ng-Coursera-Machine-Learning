from gradientFunction import gradientFunction


def gradientFunctionReg(theta, X, y, Lambda):
    """
        Compute cost and gradient for logistic regression with regularization
        computes the cost of using theta as the parameter for regularized
        logistic regression and the
        gradient of the cost w.r.t. to the parameters.
    """
    m = y.size   # number of training examples
    grad = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    grad_simple = gradientFunction(theta, X, y)
    grad = grad_simple + (Lambda*theta)/m
    # first argument without regularization
    grad[0] = grad_simple[0]

    return grad
