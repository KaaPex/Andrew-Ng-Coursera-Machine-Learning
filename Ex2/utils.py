import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    pos = X[np.where(y == 1, True, False).flatten()]
    neg = X[np.where(y == 0, True, False).flatten()]
    plt.plot(pos[:, 0], pos[:, 1], '+', markersize=7,
             markeredgecolor='black', markeredgewidth=2)
    plt.plot(neg[:, 0], neg[:, 1], 'o', markersize=7,
             markeredgecolor='black', markerfacecolor='yellow')


def mapFeature(X, degree=6):
    """
    @brief      Feature mapping function to polynomial features

    @param      X          input features
    @param      degree     number of new features

    @return     Returns a new feature array with more features, comprising of
                X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    # adds a column of ones
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    X1 = X[:, 1]
    X2 = X[:, 2]
    F = np.array([X1**(i-j)*X2**j for i in range(1, degree)
                 for j in range(i)])

    return np.hstack((X, F.T))


def plotDecisionBoundary(theta, X, y):
    """
        Plots the data points X and y into a new figure with the decision
        boundary defined by theta
        PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
        positive examples and o for the negative examples. X is assumed to be
        a either
        1) Mx3 matrix, where the first column is an all-ones column for the
        intercept.
        2) MxN, N>3 matrix, where the first column is all-ones
    """
    plt.figure()
    # plot data without fist column
    plotData(X[:, 1:], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 2]),  max(X[:, 2])])

        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                mp = mapFeature(np.array([[u[i], v[j]]]))
                z[i, j] = mp.dot(theta.T)
        # z = np.hstack((u.reshape(len(u), 1), v.reshape(len(v), 1)))
        # z = mapFeature(z)*theta

        plt.contour(u, v, z, levels=[0.0])
