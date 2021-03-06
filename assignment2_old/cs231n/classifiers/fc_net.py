from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        hid1, hid1cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, scorecache = affine_forward(hid1, self.params['W2'], self.params['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg *( np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2) )

        dhid1, grads['W2'], grads['b2'] = affine_backward(dscores, scorecache)
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dhid1, hid1cache)

        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        dimension = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, self.num_layers+1):
            self.params['W{0}'.format(i)] = weight_scale * np.random.randn(dimension[i-1], dimension[i])
            self.params['b{0}'.format(i)] = np.zeros(dimension[i])

        if self.normalization in ['batchnorm', 'layernorm']:
            self._batchnormInit()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        if not self.use_dropout:
            if self.normalization is None: # {affine-relu} X (L-1) - affine - softmax
                cache, scores = self._AffRelu_Loss(X)
            elif self.normalization is "batchnorm":
                cache, scores = self._AffBatchRelu_Loss(X)
            elif self.normalization is "layernorm":
                cache, scores = self._AffLayerRelu_Loss(X)
        else:
            if self.normalization is None: # {affine-relu} X (L-1) - affine - softmax
                cache, scores = self._AffReluDrop_Loss(X)
            elif self.normalization is "batchnorm":
                cache, scores = self._AffBatchReluDrop_Loss(X)
            elif self.normalization is "layernorm":
                cache, scores = self._AffLayerReluDrop_Loss(X)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dscores = softmax_loss(scores, y)
        if not self.use_dropout:
            if self.normalization is None: # {affine-relu} X (L-1) - affine - softmax
                grads, l2_loss = self._AffRelu_Backprop(dscores, cache)
                loss += l2_loss
            elif self.normalization is "batchnorm":
                grads, l2_loss = self._AffBatchRelu_Backprop(dscores, cache)
                loss += l2_loss
            elif self.normalization is "layernorm":
                grads, l2_loss = self._AffLayerRelu_Backprop(dscores, cache)
                loss += l2_loss
        else:
            if self.normalization is None: # {affine-relu} X (L-1) - affine - softmax
                grads, l2_loss = self._AffReluDrop_Backprop(dscores, cache)
                loss += l2_loss
            elif self.normalization is "batchnorm":
                grads, l2_loss = self._AffBatchReluDrop_Backprop(dscores, cache)
                loss += l2_loss
            elif self.normalization is "layernorm":
                grads, l2_loss = self._AffLayerReluDrop_Backprop(dscores, cache)
                loss += l2_loss
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    """
    Area For Basic Layer Settings
    """

    def _AffRelu_Loss(self, X):
        """
        loss of [{affine-relu} X (L-1) - affine] layers

        Inputs:
        - X: Array of input data of shape (N, D)

        Returns:
        - cache : cache for backprop. cache[0] is None
        cache[i] is cache from ith layer (index starts from 1)
        - scores : scores until those layers

        """
        cache = [None]
        hid = X
        for i in range(1, self.num_layers): #hidden layers
            thisW, thisb = self.params['W{0}'.format(i)], self.params['b{0}'.format(i)]
            hid, hidcache = affine_relu_forward(hid, thisW, thisb)
            cache.append(hidcache)
        #last affine
        thisW, thisb = self.params['W{0}'.format(self.num_layers)], self.params['b{0}'.format(self.num_layers)]
        scores, hidcache = affine_forward(hid, thisW, thisb)
        cache.append(hidcache)
        return cache, scores

    def _AffRelu_Backprop(self, dscores, cache):
        """
        Backpropagation of [{affine-relu} X (L-1) - affine] layers

        Inputs:
        - dscores : grad of Last layer
        - cache : cache from loss functions

        Returns:
        - grads : dictionary of grads
        - L2loss : L2 Regularization loss for other layers1
        """
        grads = {}
        loss = None
        #Last Softmax Layer
        ##Add L2 Regularization loss
        loss = 0.5 * self.reg * np.sum(self.params['W{0}'.format(self.num_layers)]**2)
        ##Calculate grads for last Affine
        dhid, grads['W{0}'.format(self.num_layers)], grads['b{0}'.format(self.num_layers)] =\
            affine_backward(dscores, cache[-1])
        grads['W{0}'.format(self.num_layers)] += self.reg * self.params['W{0}'.format(self.num_layers)]

        for i in range(self.num_layers-1, 0, -1): #hidden layers
            ##L2 Reg. loss
            loss += 0.5 * self.reg * np.sum(self.params['W{0}'.format(i)]**2)
            ##Calculate grads for [{affine-relu} X (L-1)]
            dhid, grads['W{0}'.format(i)], grads['b{0}'.format(i)] = \
                affine_relu_backward(dhid, cache[i])                   
            grads['W{0}'.format(i)] += self.reg * self.params['W{0}'.format(i)]

        return grads, loss

    
    """
    Area For Batchnorm Layer
    """

    def _batchnormInit(self):
        for i in range(1, self.num_layers):
            shape = self.params['b{0}'.format(i)].shape
            self.params['beta{0}'.format(i)] = np.zeros(shape)
            self.params['gamma{0}'.format(i)] = np.ones(shape)


    def _AffBatchRelu_Loss(self, X):
        """
        loss of [{affine-Batchnorm-relu} X (L-1) - affine] layers

        Inputs:
        - X: Array of input data of shape (N, D)

        Returns:
        - cache : cache for backprop. cache[0] is None
        cache[i] is cache Dictionary from ith layer (index starts from 1)
        - scores : scores from these layers

        """
        cache = [None]
        hid = X
        for i in range(1, self.num_layers): #hidden layers
            hidcache = {} 
            thisW, thisb = self.params['W{0}'.format(i)], self.params['b{0}'.format(i)]
            hid, hidcache['affine'] = affine_forward(hid, thisW, thisb)
            thisbeta, thisgamma = self.params['beta{0}'.format(i)], self.params['gamma{0}'.format(i)]
            hid, hidcache['batchnorm'] = batchnorm_forward(hid, thisgamma, thisbeta, self.bn_params[i-1])
            hid, hidcache['relu'] = relu_forward(hid)
            cache.append(hidcache)
        #last affine
        thisW, thisb = self.params['W{0}'.format(self.num_layers)], self.params['b{0}'.format(self.num_layers)]
        scores, hidcache = affine_forward(hid, thisW, thisb)
        cache.append(hidcache)
        return cache, scores

    def _AffBatchRelu_Backprop(self, dscores, cache):
        """
        Backpropagation of [{affine-Batchnorm-relu} X (L-1) - affine] layers

        Inputs:
        - dscores : grad of Last layer
        - cache : cache from loss functions

        Returns:
        - grads : dictionary of grads
        - L2loss : L2 Regularization loss for other layers1
        """
        grads = {}
        loss = None
        #Last Softmax Layer
        ##Add L2 Regularization loss
        loss = 0.5 * self.reg * np.sum(self.params['W{0}'.format(self.num_layers)]**2)
        ##Calculate grads for last Affine
        dhid, grads['W{0}'.format(self.num_layers)], grads['b{0}'.format(self.num_layers)] =\
            affine_backward(dscores, cache[-1])
        grads['W{0}'.format(self.num_layers)] += self.reg * self.params['W{0}'.format(self.num_layers)]

        for i in range(self.num_layers-1, 0, -1): #hidden layers
            ##L2 Reg. loss
            loss += 0.5 * self.reg * np.sum(self.params['W{0}'.format(i)]**2)
            ##Calculate grads for [{affine-Batchnorm-relu} X (L-1)]
            dhid = relu_backward(dhid, cache[i]['relu'])
            dhid, grads['gamma{0}'.format(i)], grads['beta{0}'.format(i)] = \
                batchnorm_backward_alt(dhid, cache[i]['batchnorm'])
            dhid, grads['W{0}'.format(i)], grads['b{0}'.format(i)] = \
                affine_backward(dhid, cache[i]['affine'])                   
            grads['W{0}'.format(i)] += self.reg * self.params['W{0}'.format(i)]

        return grads, loss

    """
    Area For Layernorm Layer
    """

    def _AffLayerRelu_Loss(self, X):
        """
        loss of [{affine-Layernorm-relu} X (L-1) - affine] layers

        Inputs:
        - X: Array of input data of shape (N, D)

        Returns:
        - cache : cache for backprop. cache[0] is None
        cache[i] is cache Dictionary from ith layer (index starts from 1)
        - scores : scores from these layers

        """
        cache = [None]
        hid = X
        for i in range(1, self.num_layers): #hidden layers
            hidcache = {} 
            thisW, thisb = self.params['W{0}'.format(i)], self.params['b{0}'.format(i)]
            hid, hidcache['affine'] = affine_forward(hid, thisW, thisb)
            thisbeta, thisgamma = self.params['beta{0}'.format(i)], self.params['gamma{0}'.format(i)]
            hid, hidcache['layernorm'] = layernorm_forward(hid, thisgamma, thisbeta, self.bn_params[i-1])
            hid, hidcache['relu'] = relu_forward(hid)
            cache.append(hidcache)
        #last affine
        thisW, thisb = self.params['W{0}'.format(self.num_layers)], self.params['b{0}'.format(self.num_layers)]
        scores, hidcache = affine_forward(hid, thisW, thisb)
        cache.append(hidcache)
        return cache, scores

    def _AffLayerRelu_Backprop(self, dscores, cache):
        """
        Backpropagation of [{affine-Layernorm-relu} X (L-1) - affine] layers

        Inputs:
        - dscores : grad of Last layer
        - cache : cache from loss functions

        Returns:
        - grads : dictionary of grads
        - L2loss : L2 Regularization loss for other layers1
        """
        grads = {}
        loss = None
        #Last Softmax Layer
        ##Add L2 Regularization loss
        loss = 0.5 * self.reg * np.sum(self.params['W{0}'.format(self.num_layers)]**2)
        ##Calculate grads for last Affine
        dhid, grads['W{0}'.format(self.num_layers)], grads['b{0}'.format(self.num_layers)] =\
            affine_backward(dscores, cache[-1])
        grads['W{0}'.format(self.num_layers)] += self.reg * self.params['W{0}'.format(self.num_layers)]

        for i in range(self.num_layers-1, 0, -1): #hidden layers
            ##L2 Reg. loss
            loss += 0.5 * self.reg * np.sum(self.params['W{0}'.format(i)]**2)
            ##Calculate grads for [{affine-Batchnorm-relu} X (L-1)]
            dhid = relu_backward(dhid, cache[i]['relu'])
            dhid, grads['gamma{0}'.format(i)], grads['beta{0}'.format(i)] = \
                layernorm_backward(dhid, cache[i]['layernorm'])
            dhid, grads['W{0}'.format(i)], grads['b{0}'.format(i)] = \
                affine_backward(dhid, cache[i]['affine'])                   
            grads['W{0}'.format(i)] += self.reg * self.params['W{0}'.format(i)]

        return grads, loss


    """
    Area For Dropout Layer
    """

    def _AffReluDrop_Loss(self, X):
        """
        loss of [{affine-relu-Drop} X (L-1) - affine] layers

        Inputs:
        - X: Array of input data of shape (N, D)

        Returns:
        - cache : cache for backprop. cache[0] is None
        cache[i] is cache from ith layer (index starts from 1)
        - scores : scores until those layers

        """        
        cache = [None]
        hid = X
        for i in range(1, self.num_layers): #hidden layers
            hidcache = {} 
            thisW, thisb = self.params['W{0}'.format(i)], self.params['b{0}'.format(i)]
            # print("layer {0} : hid_out - {1}, W - {2},  b - {3}".format(i, hid.shape, thisW.shape, thisb.shape))
            hid, hidcache['affine_relu'] = affine_relu_forward(hid, thisW, thisb)
            hid, hidcache['drop'] = dropout_forward(hid, self.dropout_param)
            cache.append(hidcache)
        #last affine
        thisW, thisb = self.params['W{0}'.format(self.num_layers)], self.params['b{0}'.format(self.num_layers)]
        scores, hidcache = affine_forward(hid, thisW, thisb)
        cache.append(hidcache)

        return cache, scores

    def _AffReluDrop_Backprop(self, dscores, cache):
        """
        Backpropagation of [{affine-relu} X (L-1) - affine] layers

        Inputs:
        - dscores : grad of Last layer
        - cache : cache from loss functions

        Returns:
        - grads : dictionary of grads
        - L2loss : L2 Regularization loss for other layers1
        """
        grads = {}
        loss = None
        #Last Softmax Layer
        ##Add L2 Regularization loss
        loss = 0.5 * self.reg * np.sum(self.params['W{0}'.format(self.num_layers)]**2)
        ##Calculate grads for last Affine
        dhid, grads['W{0}'.format(self.num_layers)], grads['b{0}'.format(self.num_layers)] =\
            affine_backward(dscores, cache[-1])
        grads['W{0}'.format(self.num_layers)] += self.reg * self.params['W{0}'.format(self.num_layers)]

        for i in range(self.num_layers-1, 0, -1): #hidden layers
            thiscache = cache[i]
            ##L2 Reg. loss
            loss += 0.5 * self.reg * np.sum(self.params['W{0}'.format(i)]**2)
            ##Calculate grads for [{affine-relu-drop} X (L-1)]
            dhid = dropout_backward(dhid, thiscache['drop'])
            dhid, grads['W{0}'.format(i)], grads['b{0}'.format(i)] = \
                affine_relu_backward(dhid, thiscache['affine_relu'])                   
            grads['W{0}'.format(i)] += self.reg * self.params['W{0}'.format(i)]

        return grads, loss
   

    def _AffBatchReluDrop_Loss(self, X):
        """
        loss of [{affine-Batchnorm-relu} X (L-1) - affine] layers

        Inputs:
        - X: Array of input data of shape (N, D)

        Returns:
        - cache : cache for backprop. cache[0] is None
        cache[i] is cache Dictionary from ith layer (index starts from 1)
        - scores : scores from these layers

        """
        cache = [None]
        hid = X
        for i in range(1, self.num_layers): #hidden layers
            hidcache = {} 
            thisW, thisb = self.params['W{0}'.format(i)], self.params['b{0}'.format(i)]
            hid, hidcache['affine'] = affine_forward(hid, thisW, thisb)
            thisbeta, thisgamma = self.params['beta{0}'.format(i)], self.params['gamma{0}'.format(i)]
            hid, hidcache['batchnorm'] = batchnorm_forward(hid, thisgamma, thisbeta, self.bn_params[i-1])
            hid, hidcache['relu'] = relu_forward(hid)
            hid, hidcache['drop'] = dropout_forward(hid, self.dropout_param)
            cache.append(hidcache)
        #last affine
        thisW, thisb = self.params['W{0}'.format(self.num_layers)], self.params['b{0}'.format(self.num_layers)]
        scores, hidcache = affine_forward(hid, thisW, thisb)
        cache.append(hidcache)
        return cache, scores

    def _AffBatchReluDrop_Backprop(self, dscores, cache):
        """
        Backpropagation of [{affine-Batchnorm-relu} X (L-1) - affine] layers

        Inputs:
        - dscores : grad of Last layer
        - cache : cache from loss functions

        Returns:
        - grads : dictionary of grads
        - L2loss : L2 Regularization loss for other layers1
        """
        grads = {}
        loss = None
        #Last Softmax Layer
        ##Add L2 Regularization loss
        loss = 0.5 * self.reg * np.sum(self.params['W{0}'.format(self.num_layers)]**2)
        ##Calculate grads for last Affine
        dhid, grads['W{0}'.format(self.num_layers)], grads['b{0}'.format(self.num_layers)] =\
            affine_backward(dscores, cache[-1])
        grads['W{0}'.format(self.num_layers)] += self.reg * self.params['W{0}'.format(self.num_layers)]

        for i in range(self.num_layers-1, 0, -1): #hidden layers
            ##L2 Reg. loss
            loss += 0.5 * self.reg * np.sum(self.params['W{0}'.format(i)]**2)
            ##Calculate grads for [{affine-Batchnorm-relu-drop} X (L-1)]
            dhid = dropout_backward(dhid, cache[i]['drop'])
            dhid = relu_backward(dhid, cache[i]['relu'])
            dhid, grads['gamma{0}'.format(i)], grads['beta{0}'.format(i)] = \
                batchnorm_backward_alt(dhid, cache[i]['batchnorm'])
            dhid, grads['W{0}'.format(i)], grads['b{0}'.format(i)] = \
                affine_backward(dhid, cache[i]['affine'])                   
            grads['W{0}'.format(i)] += self.reg * self.params['W{0}'.format(i)]

        return grads, loss


    def _AffLayerReluDrop_Loss(self, X):
        """
        loss of [{affine-Layernorm-relu} X (L-1) - affine] layers

        Inputs:
        - X: Array of input data of shape (N, D)

        Returns:
        - cache : cache for backprop. cache[0] is None
        cache[i] is cache Dictionary from ith layer (index starts from 1)
        - scores : scores from these layers

        """
        cache = [None]
        hid = X
        for i in range(1, self.num_layers): #hidden layers
            hidcache = {} 
            thisW, thisb = self.params['W{0}'.format(i)], self.params['b{0}'.format(i)]
            hid, hidcache['affine'] = affine_forward(hid, thisW, thisb)
            thisbeta, thisgamma = self.params['beta{0}'.format(i)], self.params['gamma{0}'.format(i)]
            hid, hidcache['layernorm'] = layernorm_forward(hid, thisgamma, thisbeta, self.bn_params[i-1])
            hid, hidcache['relu'] = relu_forward(hid)
            hid, hidcache['drop'] = dropout_forward(hid, self.dropout_param)
            cache.append(hidcache)
        #last affine
        thisW, thisb = self.params['W{0}'.format(self.num_layers)], self.params['b{0}'.format(self.num_layers)]
        scores, hidcache = affine_forward(hid, thisW, thisb)
        cache.append(hidcache)
        return cache, scores

    def _AffLayerReluDrop_Backprop(self, dscores, cache):
        """
        Backpropagation of [{affine-Layernorm-relu} X (L-1) - affine] layers

        Inputs:
        - dscores : grad of Last layer
        - cache : cache from loss functions

        Returns:
        - grads : dictionary of grads
        - L2loss : L2 Regularization loss for other layers1
        """
        grads = {}
        loss = None
        #Last Softmax Layer
        ##Add L2 Regularization loss
        loss = 0.5 * self.reg * np.sum(self.params['W{0}'.format(self.num_layers)]**2)
        ##Calculate grads for last Affine
        dhid, grads['W{0}'.format(self.num_layers)], grads['b{0}'.format(self.num_layers)] =\
            affine_backward(dscores, cache[-1])
        grads['W{0}'.format(self.num_layers)] += self.reg * self.params['W{0}'.format(self.num_layers)]

        for i in range(self.num_layers-1, 0, -1): #hidden layers
            ##L2 Reg. loss
            loss += 0.5 * self.reg * np.sum(self.params['W{0}'.format(i)]**2)
            ##Calculate grads for [{affine-Batchnorm-relu} X (L-1)]
            dhid = dropout_backward(dhid, cache[i]['drop'])
            dhid = relu_backward(dhid, cache[i]['relu'])
            dhid, grads['gamma{0}'.format(i)], grads['beta{0}'.format(i)] = \
                layernorm_backward(dhid, cache[i]['layernorm'])
            dhid, grads['W{0}'.format(i)], grads['b{0}'.format(i)] = \
                affine_backward(dhid, cache[i]['affine'])                   
            grads['W{0}'.format(i)] += self.reg * self.params['W{0}'.format(i)]

        return grads, loss

