from computeCost import computeCost
import numpy as np


def gradientDescent(X, y, theta=np.zeros((2, 1)), alpha=0.01,
                    num_iters=1500):
    """
        GRADIENTDESCENT Performs gradient descent to learn theta
        gradientDescent(X, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for _ in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        h = X.dot(theta)
        theta = theta - alpha/m*(X.T.dot(h-y))
        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history
