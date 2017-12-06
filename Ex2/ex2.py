import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


from utils import plotData
from sigmoid import sigmoid
from costFunction import costFunction
from gradientFunction import gradientFunction
from predict import predict


#  Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

#  Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = pd.read_csv('./data/ex2data1.txt', sep=',',
                   names=['test1', 'test2', 'exam'])
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values
print(data.head())

#  ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.
#
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.)')

plotData(X, y)
plt.legend(['Admitted', 'Not admitted'], loc='upper right',
           shadow=True, fontsize='x-large', numpoints=1)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
# plt.show()
input("Program paused. Press Enter to continue...")

# ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
print(X.shape)

# Add intercept term to x and X_test
X = np.hstack((np.ones((m, 1)), X))

# Initialize fitting parameters
initial_theta = np.zeros(n+1)
print(initial_theta)

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradientFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): %f' % cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

input("Program paused. Press Enter to continue...")

# ============= Part 3: Optimizing using scipy  =============
# res = minimize(costFunction, initial_theta, args=(X, y), method='BFGS',
#                jac=False, options={'disp': True, 'maxiter': 400})

res = minimize(costFunction, initial_theta, method='TNC',
               jac=False, args=(X, y),
               options={'disp': False, 'maxiter': 400})

theta = res.x
cost = res.fun

# Print theta to screen
print('Cost at theta found by scipy: %f' % cost)
print('Cost at theta of function: %f' % costFunction(theta, X, y))
print('theta:', theta)

#  ============== Part 4: Predict and Accuracies ==============
#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid(np.array([1, 45, 85]).dot(theta.T))
print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)

# Compute accuracy on our training set
y_pred = predict(theta, X)
print(y.shape)
acc = 1.0*np.where(y_pred == y)[0].size/len(y_pred) * 100
print('Train Accuracy: %f' % acc)
input("Program paused. Press Enter to continue...")
