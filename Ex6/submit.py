import numpy as np
import sys
sys.path.append('../')

import numpy as np
import scipy.io
from tasks import gaussianKernel, dataset3Params, processEmail, emailFeatures
from Submission import Submission
from Submission import sprintf


homework = 'support-vector-machines'

part_names = [
  'Gaussian Kernel',
  'Parameters (C, sigma) for Dataset 3',
  'Email Preprocessing',
  'Email Feature Extraction',
  ]

srcs = [
  'gaussianKernel.py',
  'dataset3Params.py',
  'processEmail.py',
  'emailFeatures.py',
  ]


def output(part_id):
    # Random Test Cases
    x1 = np.sin(np.arange(1, 11))
    x2 = np.cos(np.arange(1, 11))
    ec = 'the quick brown fox jumped over the lazy dog'
    wi = np.abs(np.round(x1 * 1863))
    wi = np.hstack((wi, wi)).astype('int')

    if part_id == 1:
        sim = gaussianKernel(x1, x2, 2)
        return sprintf('%0.5f ', sim)
    elif part_id == 2:
        data = scipy.io.loadmat('data/ex6data3.mat')
        X = data['X']
        y = data['y'].flatten()
        Xval = data['Xval']
        yval = data['yval'].flatten()
        C, sigma = dataset3Params(X, y, Xval, yval)
        return sprintf('%0.5f ', np.hstack((C, sigma)))
    elif part_id == 3:
        word_indices = np.array(processEmail(ec))
        return sprintf('%d', (word_indices + 1).tolist())
    elif part_id == 4:
        x = emailFeatures(wi)
        return sprintf('%d ', x)


def submit():
  s = Submission(homework, part_names, srcs, output)
  try:
      s.submit()
  except Exception as ex:
      template = "An exception of type {0} occured. Messsage:\n{1!r}"
      message = template.format(type(ex).__name__, ex.args)
      print(message)


if __name__ == "__main__":
  submit()
