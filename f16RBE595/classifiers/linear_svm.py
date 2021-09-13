from os import error
import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1 # margin of the SVM
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.transpose(X).dot(np.transpose(W))
  error = np.maximum(0, scores - (np.transpose(scores[np.arange(scores.shape[0]),y][np.newaxis, :])) + delta)
  error[np.arange(X.shape[1]),y] = 0
  loss = np.mean(np.sum(error, axis=1))
  loss += 0.5 * reg *np.sum(np.square(np.transpose(W)))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  gradient = error
  gradient[error > 0] = 1
  total_sum = np.sum(gradient, axis=1)
  gradient[np.arange(X.shape[1]), y] = -np.transpose(total_sum)
  dW = X.dot(gradient)  

  dW = np.transpose(dW)
  dW /= X.shape[1]
  dW += W*reg
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
