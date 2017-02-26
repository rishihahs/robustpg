from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as T

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

    @abstractmethod
    def sample(self, state, weights):
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

    def sample(self, state, weights):
        return np.random.choice(self.actions, p=[self.policy(state, a, weights) for a in self.actions])


"""
Normally Distributed Policy
Represents the policy as a normal pdf
"""
class GaussianPolicy(Policy):

    """
    phi: S -> R^n
    """
    def __init__(self, phi):
        self.phi = phi

        meanweights = T.dvector('meanweights')
        varweights = T.dvector('varweights')
        state = T.dvector('state')
        action = T.dscalar('action')
        mean = T.dot(meanweights, state)
        stddev = T.exp(T.dot(varweights, state))
        pdf = (1. / (stddev*np.sqrt(2*np.pi))) * T.exp(-T.sqr(action - mean) / (2*T.sqr(stddev)))

        gradlogmean = T.grad(cost=T.log(pdf), wrt=meanweights)
        gradlogvar = T.grad(cost=T.log(pdf), wrt=varweights)

        self.pdf = theano.function([state, action, meanweights, varweights], pdf)
        self.gradlogmean = theano.function([state, action, meanweights, varweights], gradlogmean)
        self.gradlogvar = theano.function([state, action, meanweights, varweights], gradlogvar)

    def policy(self, state, action, weights):
        assert(len(weights) % 2 == 0)
        s = self.phi(state)
        return self.pdf(s, action, weights[:len(weights)/2], weights[len(weights)/2:])

    def loggradient(self, state, action, weights):
        a = action[0] # Scalarify action
        s = self.phi(state)
        return np.concatenate((self.gradlogmean(s, a, weights[:len(weights)/2], weights[len(weights)/2:]),
                               self.gradlogvar(s, a, weights[:len(weights)/2], weights[len(weights)/2:])))

    def sample(self, state, weights):
        s = self.phi(state)
        return [np.random.normal(np.dot(weights[:len(weights)/2], s),
                np.exp(np.dot(weights[len(weights)/2:], s)))]

