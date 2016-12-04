from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod

"""
Abstract Mirror Descent class
"""
class MirrorDescent(object):
    __metaclass__ = ABCMeta

    """
    Valid mirror map (i.e. strictly cvx, diff'ble, etc.)
    """
    @abstractmethod
    def mirrormap(self, x):
        pass

    """
    Gradient of mirror map
    """
    @abstractmethod
    def mirrormapgrad(self, x):
        pass

    """
    Inverse of gradient of mirror map
    (i.e. gradient of Fenchel transform of mirror map)
    """
    @abstractmethod
    def mirrormapgrad_inverse(self, x):
        pass

    """
    Bregman divergence w.r.t. mirror map
    """
    def bregmandivergence(self, x, y):
        return mirrormap(x) - mirrormap(y) - np.dot(mirrormapgrad(y), x - y)

    """
    Projection onto constraint convex set
    """
    @abstractmethod
    def projection(self, x):
        pass

    """
    Mirror descent update
    grad is gradient or subgradient of function to minimize
    """
    def update(self, x, step, grad, prox=None):
        dualy = self.mirrormapgrad(x) - step*grad

        if prox is not None:
            dualy = prox(dualy)

        y = self.mirrormapgrad_inverse(dualy)
        return self.projection(y)


"""
Mirror Descent on LP Ball
"""
class LPMirrorDescent(MirrorDescent):

    def __init__(self, p):
        self.p = p
        self.q = 1. / (1. - (1. / self.p))

    def mirrormap(self, x):
        return 0.5 * np.power(np.sum(np.power(np.absolute(x), self.p)), 2. / self.p)

    def mirrormapgrad(self, x):
        return np.multiply(np.sign(x), np.power(np.absolute(x), self.q - 1)) / np.power(np.sum(np.power(np.absolute(x), self.q)), (self.q - 2.) / self.q)

    def mirrormapgrad_inverse(self, x):
        return np.multiply(np.sign(x), np.power(np.absolute(x), self.p - 1)) / np.power(np.sum(np.power(np.absolute(x), self.p)), (self.p - 2.) / self.p)

    def projection(self, x):
        return x

