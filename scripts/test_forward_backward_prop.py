from unittest import TestCase
import random
import numpy as np
from grad_check import grad_check
from forward_backward_prop import forward_backward_prop

__author__ = 'ali'


class TestForward_backward_prop(TestCase):
    def test_forward_backward_prop(self):
        N = 20
        dimensions = [10, 5, 10]
        data = np.random.randn(N, dimensions[0])  # each row will be a datum
        labels = np.zeros((N, dimensions[2]))
        for i in xrange(N):
            labels[i, random.randint(0, dimensions[2] - 1)] = 1

        params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2])

        grad_check(lambda params: forward_backward_prop(data, labels, params), params)
