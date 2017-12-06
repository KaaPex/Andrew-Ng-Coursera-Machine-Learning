import numpy as np
import sys
sys.path.append('../')

import numpy as np
from Submission import Submission
from Submission import sprintf
from net import Net

homework = 'neural-network-learning'

part_names = [
  'Feedforward and Cost Function',
  'Regularized Cost Function',
  'Sigmoid Gradient',
  'Neural Network Gradient (Backpropagation)',
  'Regularized Gradient',
  ]

srcs = [
  'nnCostFunction',
  'nnCostFunction',
  'sigmoidGradient',
  'nnCostFunction',
  'nnCostFunction',
  ]


def output(part_id):
    # Random Test Cases
    X = np.reshape(3.0*np.sin(np.linspace(1, 30, 30)), (3, 10), order='F')
    Xm = np.reshape(np.sin(np.linspace(1, 32, 32)), (16, 2), order='F')/5.0
    ym = np.array(1 + np.mod(range(1, 17), 4))
    t1 = np.sin(np.reshape(range(1, 24, 2), (4, 3), order='F'))
    t2 = np.cos(np.reshape(range(1, 40, 2), (4, 5), order='F'))

    net = Net(2, 4, 4, t1, t2)

    if part_id == 1:
        J, grad = net.costFunction(Xm, ym, 0.0)
        return sprintf('%0.5f ', J)
    elif part_id == 2:
        J, grad = net.costFunction(Xm, ym, 1.5)
        return sprintf('%0.5f ', J)
    elif part_id == 3:
        return sprintf('%0.5f ', net.sigmoidGradient(X))
    elif part_id == 4:
        J, grad = net.costFunction(Xm, ym, 0)
        grads = np.concatenate([grad['W1'].T.ravel(), grad['W2'].T.ravel()])
        return sprintf('%0.5f ', np.hstack((J, grads)).tolist())
    elif part_id == 5:
        J, grad = net.costFunction(Xm, ym, 1.5)
        grads = np.concatenate([grad['W1'].T.ravel(), grad['W2'].T.ravel()])
        return sprintf('%0.5f ', np.hstack((J, grads)).tolist())

s = Submission(homework, part_names, srcs, output)
try:
    s.submit()
except Exception as ex:
    template = "An exception of type {0} occured. Messsage:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)
