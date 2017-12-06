import numpy as np
from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape
    p = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#

# =========================================================================
    # input layer
    a1 = np.column_stack((np.ones((m, 1)), X))

    # hidden layer
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((m, 1)), a2))

    # output layer
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)

    return np.argmax(a3, axis=1)+1
