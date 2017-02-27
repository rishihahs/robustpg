from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as T

"""
Abstract Feature Generator class
"""
class FeatureGenerator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_features(self, state):
        pass


class FourierBasis(FeatureGenerator):

    """
    dimension: Dimension of state space (from S subset R^d i.e. d in (N+1)^d)
    order: Order of the Fourier Basis (i.e. N in (N+1)^d)
    """
    def __init__(self, dimension, order):

        """
        Generate coefficient matrix
        [c^1_1       ...       c^1_d]
        [            ...            ]
        [c^(n+1)^d_1 ... c^(n+1)^d_d]
        """
        self.dimension = dimension
        self.order = order
        self.C = theano.shared(self.__gen_coefficients(dimension, order), borrow=True)

        # Basis functions
        s = T.vector('s')
        phi = T.cos(np.pi*T.dot(self.C, s))
        self.phi = theano.function([s], phi, allow_input_downcast=True)

        # Linear combination with weights
        w = T.vector('w')
        self.linearcombo = theano.function([w, s], T.dot(w, phi), allow_input_downcast=True)


    """
    Generate features (basis functions) for state in R^d
    Note: state components should be normalized to range in [0, 1]
    """
    def generate_features(self, state):
        return self.phi(state)


    """
    Linear combination of weights with basis functions
    Note: state components should be normalized to range in [0, 1]
    """
    def linear_combination(self, weights, state):
        return self.linearcombo(weights, state)


    def learning_rates(self, base_learning_rate):
        norms = self.C.norm(2, axis=1).eval()
        norms[norms == 0] = 1.

        return base_learning_rate / norms


    def num_basis_functions(self):
        return (self.order + 1) ** self.dimension


    def __gen_coefficients(self, dimension, order):
        # From http://stackoverflow.com/a/34931627/603732
        def cartesian(*arrays):
            mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
            dim = len(mesh)  # number of dimensions
            elements = mesh[0].size  # number of elements, any index will do
            flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
            reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
            return reshape

        permutors = [np.arange(order+1)] * dimension
        return cartesian(*permutors)

