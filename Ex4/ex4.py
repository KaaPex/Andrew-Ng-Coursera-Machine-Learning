# Machine Learning Online Class - Exercise 4 Neural Network Learning
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from displayData import displayData
from checkNNGradients import checkNNGradients
from net import Net

# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('./data/ex4data1.mat')
X = data['X']
y = data['y']
m, _ = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

# displayData(sel)

# input("Program paused. Press Enter to continue...")

# ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
data = scipy.io.loadmat('./data/ex4weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']
y = np.squeeze(y)

# Unroll parameters
nn_params = np.hstack((Theta1.T.ravel(), Theta2.T.ravel()))
# ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
Lambda = 0.
net = Net(input_layer_size, hidden_layer_size, num_labels, Theta1, Theta2)
J, grad = net.costFunction(X, y, Lambda)

print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)\n' % J)

input("Program paused. Press Enter to continue...")

# =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
Lambda = 1.

J, _ = net.costFunction(X, y, Lambda)

print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)' % J)

# input("Program paused. Press Enter to continue...")

# ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('Evaluating sigmoid gradient...')

g = net.sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]: ')
print(g)

# input("Program paused. Press Enter to continue...")

# ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('Initializing Neural Network Parameters ...')

# net.randInitializeWeights()

print(Theta1.shape, Theta2.shape)
print(net.params['W1'].shape, net.params['W2'].shape)

# =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('Checking Backpropagation... ')

#  Check gradients by running checkNNGradients
checkNNGradients()

# input("Program paused. Press Enter to continue...")

# =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('Checking Backpropagation (Regularization) ... ')

#  Check gradients by running checkNNGradients
Lambda = 3.0
# checkNNGradients(Lambda)

# Also output the costFunction debugging values
debug_J, _ = net.costFunction(X, y, Lambda)

print('Cost at (fixed) debugging parameters (lambda = 3): %f (this value should be about 0.576051)\n\n' % debug_J)

# input("Program paused. Press Enter to continue...")

# =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('Training Neural Network... ')

# After you have completed the assignment, change the MaxIter to a larger
# value to see how more training helps.
# options = optimset('MaxIter', 50)

#  You should also try different values of lambda
Lambda = 1

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
# res = net.train(X_train, y_train, X_test, y_test,
#                 reg=1.0, batch_size=200, num_iters=1000, verbose=True)

# input("Program paused. Press Enter to continue...")

#
# ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

print('Visualizing Neural Network... ')

displayData(net.params['W1'][:, 1:])

# plt.plot(res['val_acc_history'])
# plt.show()

# input("Program paused. Press Enter to continue...")

# ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = net.predict(X)

accuracy = np.mean(np.double(pred == y)) * 100
print('Training Set Accuracy: %f\n' % accuracy)

# input("Program paused. Press Enter to exit...")
