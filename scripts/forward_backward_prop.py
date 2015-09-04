import numpy as np

from sigmoid import sigmoid, sigmoid_grad
from softmax import softmax

def initialize_weights(dimensions = [10,5,10]):
    params = np.array([])
    for i in xrange(len(dimensions) - 1):
        w = np.random.randn(dimensions[i] * dimensions[i + 1])
        b = np.random.randn(dimensions[i + 1])
        s = np.sqrt(6) / np.sqrt(dimensions[i] + dimensions[i + 1])
        w = w * 2 * s - s
        params = np.append(params, w)
        params = np.append(params, b)

    return params


def forward_backward_prop(data, labels, params, dimensions = [10, 5, 10]):
    """ Forward and backward propagation for a two-layer sigmoidal network """
    ###################################################################
    # Compute the forward propagation and for the cross entropy cost, #
    # and backward propagation for the gradients for all parameters.  #
    ###################################################################

    # Unpack network parameters (do not modify)
    t = 0
    W1 = np.reshape(params[t:t + dimensions[0] * dimensions[1]], (dimensions[0], dimensions[1]))
    t += dimensions[0] * dimensions[1]
    b1 = np.reshape(params[t:t + dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t + dimensions[1] * dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1] * dimensions[2]
    b2 = np.reshape(params[t:t + dimensions[2]], (1, dimensions[2]))

    # YOUR CODE HERE: forward propagation
    A1 = np.dot(data, W1) + b1  # broadcasting on b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2) + b2
    Yhat = softmax(A2)  # Y output

    # cost = ...
    index = labels == 1
    logYhat = np.log(Yhat)
    cost = -np.sum(logYhat[index])
    # END YOUR CODE

    # YOUR CODE HERE: backward propagation
    targets = np.zeros(np.shape(Yhat))
    targets[index] = 1
    dA2 = Yhat - targets
    db2 = sum(dA2)
    dW2 = np.dot(Z1.T, dA2)

    dZ1 = dA2.dot(W2.T)
    dA1 = np.multiply(sigmoid_grad(Z1), dZ1)
    db1 = sum(dA1)
    dW1 = np.dot(data.T, dA1)

    gradb1 = db1
    gradW1 = dW1
    gradb2 = db2
    gradW2 = dW2
    # END YOUR CODE

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad
