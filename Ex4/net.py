import numpy as np
import pandas as pd


class Net():
    def __init__(self, input_size, hidden_size, output_size, w1, w2):
        self.params = {}
        self.params['W1'] = w1
        self.params['W2'] = w2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def sigmoid(self, z):
        """
        @brief      computes the sigmoid of z

        @param      z     z can be a matrix, vector or scalar

        @return     sigmoid
        """

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the sigmoid of each value of z (z can be a
        # matrix, vector or scalar).
        return 1/(1+np.exp(-z))

    def sigmoidGradient(self, z):
        """
        @brief      computes the gradient of sigmoid function

        @param      z     z can be a matrix, vector or scalar

        @return     sigmoid
        """

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the sigmoid of each value of z (z can be a
        # matrix, vector or scalar).
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def randInitializeWeights(self, std=1e-4):
        """randomly initializes the weights of a layer with L_in
          incoming connections and L_out outgoing connections.
          Note that W should be set to a matrix of size(L_out, 1 + L_in)
          as the column row of W handles the "bias" terms
        """

        # ====================== YOUR CODE HERE ======================
        # Instructions: Initialize W randomly so that
        # we break the symmetry while
        # training the neural network.
        #
        # Note: The first row of W corresponds to the parameters
        # for the bias units
        #

        epsilon_init = np.sqrt(6)/np.sqrt(self.input_size + self.output_size)
        W1_shape = (self.hidden_size, self.input_size+1)
        W2_shape = (self.output_size, self.hidden_size+1)
        self.params['W1'] = np.random.randn(*W1_shape)
        self.params['W1'] *= 2 * epsilon_init - epsilon_init

        self.params['W2'] = np.random.randn(*W2_shape)
        self.params['W2'] *= 2 * epsilon_init - epsilon_init

    def computeNumericalGradient(self, X, y, Lambda=0, e=1e-4):
        """computes the numerical gradient of the function J around theta.
        Calling y = J(theta) should return the function value at theta.
        """
# Notes: The following code implements numerical gradient checking, and
#        returns the numerical gradient.It sets numgrad(i) to (a numerical
#        approximation of) the partial derivative of J with respect to the
#        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
#        be the (approximately) the partial derivative of J with respect
#        to theta(i).)

        numgrad = {}
        params = self.params

        for key, W in params.items():
            m, n = W.shape
            numgrad[key] = np.zeros(W.shape)
            for i in range(m):
                for j in range(n):
                    self.params[key] = params[key]
                    self.params[key][i, j] -= e
                    loss1, _ = self.costFunction(X, y, Lambda)
                    self.params[key] = params[key]
                    self.params[key][i, j] += e
                    loss2, _ = self.costFunction(X, y, Lambda)
                    numgrad[key][i, j] = (loss2 - loss1) / (2*e)

        self.params = params

        return numgrad

    def costFunction(self, X, y, reg=0.0):
        """
        @brief      compute cost function and gradien

        @param      X       Input training data of shape(N,D)
        @param      y       Vector of training labels
        @param      Lambda  Regularisation strength

        @return     cost function and gradient for two layer network
        """
        m, _ = X.shape
        W1 = self.params['W1']
        W2 = self.params['W2']

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial
#         derivatives of the cost function with respect to Theta1 and
#         Theta2 in Theta1_grad
#         and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#
        J = 0
        grads = {}

        # a1
        # 5000x400 => 5000x401
        a1 = np.column_stack((np.ones((m, 1)), X))

        # compute scores
        # z2
        # print(X.shape, a1.shape, W1.shape)
        z2 = a1.dot(W1.T)  # 5000x401 * 401x25 = 5000x25

        # a2
        # activation for hidden layer
        a2 = self.sigmoid(z2)
        a2 = np.column_stack((np.ones((m, 1)), a2))  # 5000x26

        # z3
        z3 = a2.dot(W2.T)  # 5000x26 * 26x10 = 5000x10

        # a3
        # activation for hidden layer
        # 5000x10
        a3 = self.sigmoid(z3)

        # One Hot Encoding for y
        y_matrix = pd.get_dummies(y).as_matrix()

        # cost function without regularisation
        J = -np.sum(y_matrix*np.log(a3) +
                    (1-y_matrix)*np.log(1-a3))
        J /= m
        # add regularisation
        J += 0.5*reg*(np.sum(np.square(W1[:, 1:])) +
                      np.sum(np.square(W2[:, 1:])))/m

        # Backpropagation
        # Compute gradient (back propagation)
        # d3 = a3 - y_ohe
        # 5000x10 - 5000x10 = 5000x10
        sigma3 = a3 - y_matrix

        # d2 = Theta2*d3.*sig_grad(z2)
        # 5000x10*10x26 = 5000x26
        # 5000x26 => 5000x25
        # 5000x25*5000x25 = 5000x25
        sigma2 = sigma3.dot(W2)
        sigma2 = sigma2[:, 1:]  # skip first column
        sigma2 *= self.sigmoidGradient(z2)

        # print(a3.shape, sigma3.shape, W2.shape)
        # delta2 = sigma3*a2/m
        # 10x5000*5000x26 = 10x26
        grads['W2'] = sigma3.T.dot(a2)/m
        # print(W2.shape, grads['W2'].shape)

        # d1 = sigma2*(a1=X)/m
        # 25x5000*5000x401 = 401x25
        grads['W1'] = sigma2.T.dot(a1)/m
        # print(W1.shape, grads['W1'].shape)

        # add regularization for gradients, skip first
        grads['W1'][:, 1:] += reg*W1[:, 1:]/m
        grads['W2'][:, 1:] += reg*W2[:, 1:]/m

        return J, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):

        # self.randInitializeWeights()
        self.cost_history = []

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]

            loss, grads = self.costFunction(X_batch, y_batch, reg)

            self.params['W1'] += -learning_rate * grads['W1']
            self.params['W2'] += -learning_rate * grads['W2']

            loss_history.append(loss)

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and
            # decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """ outputs the predicted label of X given the
        trained weights of a neural network (Theta1, Theta2)
        """

        # Useful values
        m, _ = X.shape
        W1 = self.params['W1']
        W2 = self.params['W2']
        # ====================== YOUR CODE HERE ======================
        # Instructions: Complete the following code to make predictions using
        #               your learned neural network. You should set p to a
        #               vector containing labels between 1 to num_labels.
        #
        # Hint: The max function might come in useful. In particular, the max
        #       function can also return the index of the max element, for more
        #       information see 'help max'. If your examples are in rows,
        #        then, you
        #       can use max(A, [], 2) to obtain the max for each row.
        #
        # ====================================================================
        # input layer
        a1 = np.column_stack((np.ones((m, 1)), X))

        # hidden layer
        z2 = a1.dot(W1.T)
        a2 = self.sigmoid(z2)
        a2 = np.column_stack((np.ones((m, 1)), a2))

        # output layer
        z3 = a2.dot(W2.T)
        a3 = self.sigmoid(z3)

        return np.argmax(a3, axis=1)+1
