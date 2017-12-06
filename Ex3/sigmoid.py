import numpy as np


def sigmoid(z):
    """
    @brief      computes the sigmoid of z

    @param      z     z can be a matrix, vector or scalar

    @return     sigmoid
    """
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).
    return 1/(1+np.exp(-z))
