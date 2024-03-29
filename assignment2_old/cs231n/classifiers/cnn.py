from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, debug=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        C, H, W = input_dim
        #conv layer shape
        pad =  (filter_size - 1) // 2
        convH = int( 1 + (H + 2 * pad - filter_size))
        convW = int( 1 + (W + 2 * pad - filter_size))
        if debug:
            print(pad, filter_size, convH, convW)
        #Pool Layer Size
        stride = 2
        pool_height = 2
        pool_width = 2
        poolH = int(1 + (convH -pool_height) / stride)
        poolW = int(1 + (convW - pool_width) / stride)
        if debug:
            print(poolH, poolW)

        input_dimension = num_filters * poolH * poolW

        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        dimension = [None, input_dimension, hidden_dim, num_classes]
        if debug:
            print(dimension)
        for i in range(2, 4):
            #for 2/3th affine layer
            self.params['W{0}'.format(i)] = weight_scale * np.random.randn(dimension[i-1], dimension[i])
            self.params['b{0}'.format(i)] = np.zeros(dimension[i])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None, debug = False):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        #conv - relu - 2x2 max pool - affine - relu - affine - softmax
        cache = {}
        if debug:
            for i in self.params.keys():
                print(i, self.params[i].shape)
        scores, cache['conv'] = conv_forward_fast(X, W1, b1, conv_param)
        if debug:
            print('score_Conv : ', scores.shape)
        scores, cache['relu'] = relu_forward(scores)
        if debug:
            print('score_Relu : ', scores.shape)
        scores, cache['pool'] = max_pool_forward_fast(scores, pool_param)
        if debug:
            print('score_PoolI : ', scores.shape)
        scores, cache['aff1relu'] = affine_relu_forward(scores, W2, b2)
        if debug:
            print('score_AffIRelu : ', scores.shape)
        scores, cache['aff2'] = affine_forward(scores, W3, b3)
        if debug:
            print('score_AffII : ', scores.shape)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        for i in range(1,4):
            grads['W{0}'.format(i)] = 0
            grads['b{0}'.format(i)] = 0
        loss, dscores = softmax_loss(scores, y)
        for i in range(1, 4): #L2 Loss
            loss += 0.5 * self.reg * np.sum(self.params['W{0}'.format(i)]**2)
            grads['W{0}'.format(i)] += self.reg * self.params['W{0}'.format(i)]
        
        dhid, dw, db = affine_backward(dscores, cache['aff2'])
        grads['W3'] += dw 
        grads['b3'] += db
        dhid, dw, db = affine_relu_backward(dhid, cache['aff1relu'])
        grads['W2'] += dw 
        grads['b2'] += db
        dhid = max_pool_backward_fast(dhid, cache['pool'])
        dhid = relu_backward(dhid, cache['relu'])
        dx, dw, db  = conv_backward_fast(dhid, cache['conv'])
        grads['W1'] += dw 
        grads['b1'] += db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
