from unittest import TestCase
from grad_check import grad_check
from softmax import softmax
import numpy as np

__author__ = 'ali'


class TestGrad_check(TestCase):
    def test_grad_check(self):
        quad = lambda x: (np.sum(x ** 2), x * 2)

        grad_check(quad, np.array(123.456))  # scalar test
        grad_check(quad, np.random.randn(3, ))  # 1-D test
        grad_check(quad, np.random.randn(4, 5))  # 2-D test





