import scipy.io
import numpy as np
from gradientFunctionReg import gradientFunctionReg
from gradientFunction import gradientFunction
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from displayData import displayData


# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10         # 10 labels, from 1 to 10
                        # (note that we have mapped "0" to label 10

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = scipy.io.loadmat('./data/ex3data1.mat')  # training data stored in arrays X, y
X = data['X']
y = data['y'].T[0]
m, _ = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

# displayData(sel)

input('Program paused. Press enter to continue.\n')

# ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#
#
# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.concatenate((np.ones((5, 1)),
                      np.fromiter((x/10 for x in range(1, 16)), float)
                      .reshape((3, 5)).T), axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
J = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = gradientFunctionReg(theta_t, X_t, y_t, lambda_t)

print('Cost: %f' % J)
print('Expected cost: 2.534819\n')
print('Gradients:')
print(grad)
print('Expected gradients:')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

print('Program paused. Press enter to continue.\n')

# ============ Part 2b: One-vs-All Training ============
print('Training One-vs-All Logistic Regression...')

Lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, Lambda)

input("Program paused. Press Enter to continue...")

# ================ Part 3: Predict for One-Vs-All ================
#  After ...
pred = predictOneVsAll(all_theta, X)
print(pred)
accuracy = np.mean(np.double(pred == y.ravel())) * 100
print('\nTraining Set Accuracy: %f\n' % accuracy)
