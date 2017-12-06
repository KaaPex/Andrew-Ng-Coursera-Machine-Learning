import numpy as np
from sigmoid import sigmoid


def predictOneVsAll(all_theta, X):
    """will return a vector of predictions
  for each example in the matrix X. Note that X contains the examples in
  rows. all_theta is a matrix where the i-th row is a trained logistic
  regression theta vector for the i-th class. You should set p to a vector
  of values from 1..K (e.g., p = [1 3 1 2] predicts classes 1, 3, 1, 2
  for 4 examples) """

    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters (one-vs-all).
#               You should set p to a vector of predictions (from 1 to
#               num_labels).
#
# Hint: This code can be done all vectorized using the max function.
#       In particular, the max function can also return the index of the
#       max element, for more information see 'help max'. If your examples
#       are in rows, then, you can use max(A, [], 2) to obtain the max
#       for each row.
#
    hypoths = sigmoid(X.dot(all_theta.T))

# Adding one because Python uses zero based indexing for the 10 columns (0-9),
# while the 10 classes are numbered from 1 to 10.
    return (np.argmax(hypoths, axis=1) + 1)
