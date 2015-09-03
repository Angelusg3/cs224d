from unittest import TestCase
from softmax import softmax
import numpy as np

__author__ = 'ali'


class TestSoftmax(TestCase):
    def test_softmax_2d(self):
        print softmax(np.array([[1,2,3],
                               [12,32,10]]))
    def test_softmax_1d(self):
        print softmax(np.array([1,2,3]).T)