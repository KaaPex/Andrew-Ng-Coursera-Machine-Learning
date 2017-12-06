# Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


def polyFeatures(X, p):
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


def plotFit(min_x, max_x, mu, sigma, theta, p):
    """plots the learned polynomial fit with power p
    and feature normalization (mu, sigma).
    """

# We plot a range slightly bigger than the min and max values to get
# an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).T

# Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

# Add ones
    X_poly = np.column_stack((np.ones(x.shape[0]), X_poly))

# Plot
    plt.plot(x, X_poly.dot(theta), '--', lw=2)
    plt.show()


def featureNormalize(X):
    """ returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """

    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma

    return X_norm, mu, sigma

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = scipy.io.loadmat('./data/ex5data1.mat')

# m = Number of examples
X = data['X'][:, 0]
y = data['y'][:, 0]
Xval = data['Xval'][:, 0]
yval = data['yval'][:, 0]
Xtest = data['Xtest'][:, 0]

m = X.size

# Plot training data
plt.scatter(X, y, marker='x', s=60, edgecolor='r', lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')  # Set the y-axis label
plt.xlabel('Change in water level (x)')  # Set the x-axis label
# plt.show()

input("Program paused. Press Enter to continue...")

# =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.
#
lr = LinearRegression()

theta = np.array([1, 1])
J, grad = lr.costFunction(np.column_stack((np.ones(m), X)), y, theta, 1)

print('Cost at theta = [1  1]: %f \n(this value should be about 303.993192)\n' % J)

input("Program paused. Press Enter to continue...")

# =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.
#

print('Gradient at theta = [1  1]:  [%f %f] \n(this value should be about [-15.303016 598.250744])\n' %(grad[0], grad[1]))

input("Program paused. Press Enter to continue...")

# =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.
#

#  Train linear regression with Lambda = 0
Lambda = 0
lr = LinearRegression()
theta = lr.fit(np.column_stack((np.ones(m), X)), y, reg=Lambda)

#  Plot fit over the data
plt.scatter(X, y, marker='x', s=20, edgecolor='r', lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')  # Set the y-axis label
plt.xlabel('Change in water level (x)')  # Set the x-axis label
plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), '--', lw=2.0)
# plt.show()

# input("Program paused. Press Enter to continue...")

# =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
#

Lambda = 0
error_train, error_val = lr.learningCurve(np.column_stack((np.ones(m), X)), y,
                                       np.column_stack((np.ones(Xval.shape[0]),
                                                        Xval)), yval, Lambda)
plt.figure()
plt.plot(np.arange(0, m), error_train,
         color='b', lw=0.5, label='Train')
plt.plot(np.arange(0, m), error_val,
         color='r', lw=0.5, label='Cross Validation')
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')

plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.show()

print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

input("Program paused. Press Enter to continue...")

# =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones(m), X_poly))  # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_test))  # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_val))  # Add Ones

print('Normalized Training Example 1:')
print(X_poly[0, :])

input('\nProgram paused. Press enter to continue.')

# =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of Lambda. The code below runs polynomial regression with
#  Lambda = 0. You should try running the code with different values of
#  Lambda to see how the fit and learning curve change.
#


Lambda = 1
lr = LinearRegression()
theta = lr.fit(X_poly, y, Lambda)

# Plot training data and fit
plt.figure()
plt.scatter(X, y, marker='x', s=10, edgecolor='r', lw=1.5)

plotFit(min(X), max(X), mu, sigma, theta, p)

plt.xlabel('Change in water level (x)')            # Set the y-axis label
plt.ylabel('Water flowing out of the dam (y)')     # Set the x-axis label
# plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), marker='_',  lw=2.0)
plt.title('Polynomial Regression Fit (Lambda = %f)' % Lambda)

error_train, error_val = lr.learningCurve(X_poly, y, X_poly_val, yval, Lambda)
plt.plot(np.arange(1, 13), error_train, label='Train')
plt.plot(np.arange(1, 13), error_val, label='Cross Validation')
plt.title('Polynomial Regression Learning Curve (Lambda = %f)' % Lambda)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend()
plt.show()

print('Polynomial Regression (Lambda = %f)\n\n' % Lambda)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

input("Program paused. Press Enter to continue...")

# =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  Lambda on a validation set. You will then use this to select the
#  "best" Lambda value.
#

Lambda_vec, error_train, error_val = lr.validationCurve(X_poly, y,
                                                        X_poly_val, yval)
plt.figure()
plt.plot(Lambda_vec, error_train, Lambda_vec, error_val)
plt.legend('Train', 'Cross Validation')
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.show()

print('Lambda\t\tTrain Error\tValidation Error')
for i in range(Lambda_vec.size):
    print(' %f\t%f\t%f' % (Lambda_vec[i], error_train[i], error_val[i]))

input("Program paused. Press Enter to continue...")
