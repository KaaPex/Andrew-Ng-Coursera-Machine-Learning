import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import plotData, mapFeature, plotDecisionBoundary
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from scipy.optimize import minimize

#  Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


def optimize(theta, X, y, Lambda):

    result = minimize(costFunctionReg, theta, method='L-BFGS-B',
                      jac=gradientFunctionReg, args=(X, y, Lambda),
                      options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

    return result


data2 = pd.read_csv('./data/ex2data2.txt', sep=',',
                    names=['test1', 'test2', 'y'])
X = data2.iloc[:, :2].values
y = data2.iloc[:, 2].values
print(data2.head())

#plotData(X, y)
plt.legend(['y = 1', 'y = 0'], loc='upper right',
           shadow=True, fontsize='x-large', numpoints=1)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
#plt.show()
input("Program paused. Press Enter to continue...")

# =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X)
print(X.shape)

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
Lambda = 1.

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, Lambda)

print('Cost at initial theta (zeros): %f\n', cost)
print('Expected cost (approx): 0.693\n')

grad = gradientFunctionReg(initial_theta, X, y, Lambda)
print('Gradient at initial theta (zeros) - first five values only:\n')
print(grad)
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

input('\nProgram paused. Press enter to continue.\n')

# ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
Lambda = 1.
result = optimize(initial_theta, X, y, Lambda)
theta = result.x
cost = result.fun
# Print theta to screen
print('Lambda: %f', Lambda)
print('Cost at theta found by scipy: %f' % cost)
print('Theta:', theta)

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.title(r'$\lambda$ = ' + str(Lambda))

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()
