__author__ = 'ali'
import numpy as np
import random
def grad_check(f, x):
    """
    Gradient check for the given function
    :param f: it should be a function that takes a single argument and returns the value and its gradient
    :param x: the point that we check the gradient
    :return: pass of fail
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)
    h = 1e-4

    # Iterate over all indexes in point
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # calculate numerical gradient
        xph = np.copy(x)
        xph[ix] = xph[ix] + h
        random.setstate(rndstate)
        fph, tmp = f(xph)
        xmh = np.copy(x)
        xmh[ix] = xmh[ix] - h
        random.setstate(rndstate)
        fmh, tmp = f(xmh)
        numgrad = (fph - fmh) / (2.0 * h)

        diff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if (diff > 1e-5):
            print "Gradient check failed."
            print "Function gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return False

        it.iternext()

    print "Gradient check passed!"
    return True

