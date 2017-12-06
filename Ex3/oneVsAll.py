import numpy as np
from scipy.optimize import minimize, fmin_cg
from lrCostFunction import lrCostFunction
from gradientFunctionReg import gradientFunctionReg


def optimize(theta, X, y, Lambda):
    """
    result = minimize(lrCostFunction, theta, method='L-BFGS-B',
                      jac=gradientFunctionReg, args=(X, y, Lambda),
                      options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

    result = minimize(lrCostFunction, theta, method='TNC',
                      jac=gradientFunctionReg, args=(X, y, Lambda),
                      options={'disp': False, 'maxiter': 400})

    result = fmin_cg(lrCostFunction, fprime=gradientFunctionReg,
                     x0=theta, args=(X, y, Lambda), maxiter=50,
                     disp=False, full_output=True)
    """
    result = minimize(lrCostFunction, theta, method='L-BFGS-B',
                      jac=gradientFunctionReg, args=(X, y, Lambda),
                      options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})
    return result


def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n+1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda.
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

# Example Code for fmincg:
    # Set Initial theta
    initial_theta = np.zeros(n+1)

    # This function will return theta and the cost
    # optimise theta for each labels
    """
    y_label = (y == 1).astype(int).T[0]
    cost = lrCostFunction(initial_theta, X, y_label, Lambda)
    print(cost.shape)
    print('Cost at initial theta (zeros): %f\n' % cost)

    grad = gradientFunctionReg(initial_theta, X, y_label, Lambda)
    print(grad.shape)
    print('Gradient at initial theta (zeros) - first five values only:\n')
    print(grad)
    """

    for c in np.arange(1, num_labels+1):
        y_label = (y == c).astype(int).T[0]
        res = optimize(initial_theta, X, y_label, Lambda)
        all_theta[c-1] = res.x

    return all_theta
