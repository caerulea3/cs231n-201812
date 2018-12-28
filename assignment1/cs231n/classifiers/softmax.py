import numpy as np
from random import shuffle
from tqdm import tqdm_notebook as tqdm
from copy import deepcopy as deepcopy

def softmax_loss_naive(W, X, y, reg, verbose=False):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization!                                                            #
    #############################################################################
    scores = np.dot(X, W)
    print(scores.shape)
    train_num=X.shape[0]
    img_size = X.shape[1]
    category_num = W.shape[1]
    
    for i in range(train_num):
        score_exp = np.exp(scores[i] - scores[i].max()) 
        score_exp_sum = score_exp.sum()
        softmax = score_exp / score_exp_sum
        loss += - np.log (softmax[y[i]]) / train_num    
    loss += np.sum(W**2) * reg
    
    for i in range(img_size) if not verbose else tqdm(range(img_size)):
        score = scores[i]
        score_exp = np.exp(scores[i] - scores[i].max())
        score_exp_sum = score_exp.sum()
        softmax = score_exp / score_exp_sum
        for j in range(category_num):
            if j == y[i]:
                dW[i, j] = score[j] * ((score_exp[j] / score_exp_sum) - 1)
            else:
                dW[i, j] = -1 * score_exp[j] * X[i, j] / softmax[y[i]]
    #############################################################################
    #                           END OF YOUR CODE                                #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                           END OF YOUR CODE                                #
    #############################################################################

    return loss, dW

