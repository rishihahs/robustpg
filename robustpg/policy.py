from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod

"""
Abstract (Differentiable) Policy class
"""
class Policy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def policy(self, state, action, weights):
        pass

    @abstractmethod
    def loggradient(self, state, action, weights):
        pass


"""
Linear softmax policy
TODO: Generalize to nonlinear combinations
"""
class SoftmaxPolicy(Policy):

    """
    actions: all possible actions
    phi: (s, a) -> R^n: features state, action pair
    """
    def __init__(self, actions, phi):
        self.actions = actions
        self.phi = phi

    def policy(self, state, action, weights):
        total = np.sum([np.exp(np.dot(weights, self.phi(state, b))) for b in self.actions])
        return np.exp(np.dot(weights, self.phi(state, action))) / total

    def loggradient(self, state, action, weights):
        # Sutton's softmax gradient
        # return self.phi(state, action) - np.sum([self.policy(state, b, weights)*self.phi(state, b) for b in self.actions])

        # My softmax gradient
        total = np.sum([np.exp(np.dot(weights, self.phi(state, b))) for b in self.actions])
        return self.phi(state, action)/total - np.sum([self.policy(state, b, weights)*self.phi(state, b) for b in self.actions])

