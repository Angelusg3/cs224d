import numpy as np
import random
from softmax import softmax

# Set up fake data and parameters for the neural network
N = 20
dimensions = [10, 5, 10]
data = np.random.randn(N, dimensions[0])   # each row will be a datum
labels = np.zeros((N, dimensions[2]))
for i in xrange(N):
    labels[i,random.randint(0,dimensions[2]-1)] = 1

params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

def sigmoid(x):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################

    ### YOUR CODE HERE
    x = 1. / np.exp(np.negative(x))
    ### END YOUR CODE

    return x

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    ###################################################################
    # Compute the gradient for the sigmoid function here. Note that   #
    # for this implementation, the input f should be the sigmoid      #
    # function value of your original input x.                        #
    ###################################################################

    ### YOUR CODE HERE
    f = f * (1 - f)
    ### END YOUR CODE

    return f

def forward_backward_prop(data, labels, params):
    """ Forward and backward propagation for a two-layer sigmoidal network """
    ###################################################################
    # Compute the forward propagation and for the cross entropy cost, #
    # and backward propagation for the gradients for all parameters.  #
    ###################################################################

    ### Unpack network parameters (do not modify)
    t = 0
    W1raw = params[t:t + dimensions[0] * dimensions[1]]
    W1 = np.reshape(W1raw, (dimensions[0], dimensions[1]))
    t += dimensions[0]*dimensions[1]
    b1 = np.reshape(params[t:t+dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t+dimensions[1]*dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1]*dimensions[2]
    b2 = np.reshape(params[t:t+dimensions[2]], (1, dimensions[2]))

    ### YOUR CODE HERE: forward propagation
    A1 = np.dot(data, W1) + b1    # broadcasting on b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2) + b2
    Yhat = softmax(A2)   # Y output

    # cost = ...
    index = labels == 1
    logYhat = np.log(Yhat)
    cost = -np.sum(logYhat[index])
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    targets = np.zeros(np.shape(Yhat))
    targets[index] = 1
    dA2 = Yhat - targets
    db2 = sum(dA2)
    dW2 = np.dot(Z1.T, dA2)

    dZ1 = dA2.dot(W2.T)
    dA1 = np.multiply(sigmoid_grad(A1), dZ1)
    db1 = sum(dA1)
    dW1 = np.dot(data.T, dA1)

    gradb1 = db1
    gradW1 = dW1
    gradb2 = db2
    gradW2 = dW2
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad


