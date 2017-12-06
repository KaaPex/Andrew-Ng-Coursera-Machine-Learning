import numpy as np
from scipy.optimize import minimize


class LinearRegression:
    def __init__(self):
        pass

    def costFunction(self, X, y, W, reg):
        m = X.shape[0]

        J = 0
        grad = np.zeros(W.shape[0])

        h = X.dot(W)
        J = 0.5*np.sum((h-y)**2)/m

        # add regularisation? skip zero
        J += 0.5*reg*np.sum(W[1:]**2)/m

        grad = (h-y).dot(X)/m
        # add reg to grad, skip zero
        grad[1:] += reg*W[1:]/m

        return J, grad

    def learningCurve(self, X, y, Xval, yval, reg):
        """returns the train and
    cross validation set errors for a learning curve. In particular,
    it returns two vectors of the same length - error_train and
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).
    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.
        """

        # Number of training examples
        m, _ = X.shape
        # You need to return these values correctly
        error_train = np.zeros((m, 1))
        error_val = np.zeros((m, 1))

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return training errors in
#               error_train and the cross validation errors in error_val.
#               i.e., error_train(i) and
#               error_val(i) should give you the errors
#               obtained after training on i examples.
#
# Note: You should evaluate the training error on the first i training
#       examples (i.e., X(1:i, :) and y(1:i)).
#
#       For the cross-validation error, you should instead evaluate on
#       the _entire_ cross validation set (Xval and yval).
#
# Note: If you are using your cost function (linearRegCostFunction)
#       to compute the training and cross validation error, you should
#       call the function with the lambda argument set to 0.
#       Do note that you will still need to use lambda when running
#       the training to obtain the theta parameters.
#
# Hint: You can loop over the examples with the following:
#
#       for i = 1:m
#           # Compute train/cross validation errors using training examples
#           # X(1:i, :) and y(1:i), storing the result in
#           # error_train(i) and error_val(i)
#           ....
#       end
#
        for i in np.arange(m):
            X_t = X[:i+1, :]
            y_t = y[:i+1]
            W_t = self.fit(X_t, y_t, reg)
            error_train[i], _ = self.costFunction(X_t, y_t, W_t, 0)
            error_val[i], _ = self.costFunction(Xval, yval, W_t, 0)

        return error_train, error_val

    def polyFeatures(self, X, p):
        """takes a data matrix X (size m x 1) and
        maps each example into its polynomial features where
        X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
        """
        # You need to return the following variables correctly.
        X = X.reshape(-1, 1)
        X_poly = X.copy()
    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th
    #               column of X contains the values of X to the p-th power.
    #
        for i in range(p-1):
            dim = i+2
            X_poly = np.append(X_poly, np.power(X, dim), axis=1)

        return X_poly

    def validationCurve(self, X, y, Xval, yval):
        """returns the train
        and validation errors (in error_train, error_val)
        for different values of lambda. You are given the training set (X,
        y) and validation set (Xval, yval).
        """

    # Selected values of lambda (you should not change this)
        lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03,
                              0.1, 0.3, 1, 3, 10])

    # You need to return these variables correctly.
        error_train = np.zeros(lambda_vec.size)
        error_val = np.zeros(lambda_vec.size)

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return training errors in
#               error_train and the validation errors in error_val. The
#               vector lambda_vec contains the different lambda parameters
#               to use for each calculation of the errors, i.e,
#               error_train(i), and error_val(i) should give
#               you the errors obtained after training with
#               lambda = lambda_vec(i)
#
# Note: You can loop over lambda_vec with the following:
#
#       for i = 1:length(lambda_vec)
#           lambda = lambda_vec(i)
#           # Compute train / val errors when training linear
#           # regression with regularization parameter lambda
#           # You should store the result in error_train(i)
#           # and error_val(i)
#           ....
#       end
#
#
        for idx, reg in enumerate(lambda_vec):
            W_t = self.fit(X, y, reg)
            error_train[idx], _ = self.costFunction(X, y, W_t, 0)
            error_val[idx], _ = self.costFunction(Xval, yval, W_t, 0)

        return lambda_vec, error_train, error_val

    def fit(self, X, y, reg=0, method='CG', maxiter=200):

        """trains linear regression using
        the dataset (X, y) and regularization parameter lambda. Returns the
        trained parameters theta.
        """

    # Initialize Theta
        initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
        costFunc = lambda t: self.costFunction(X, y, t, reg)[0]

        result = minimize(costFunc, initial_theta, method=method, jac=None,
                          options={'disp': False, 'maxiter': maxiter})

        return result.x
