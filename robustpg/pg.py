from __future__ import division
import random
import math
import numpy as np

from mirrordescent import LPMirrorDescent

class REINFORCE(object):

    def __init__(self, policy, actions, initial_weights, update, regularization_param=0.0):
        self.policy = policy
        self.actions = actions
        self.weights = initial_weights
        self.gamma = 1.
        self.alpha = 0.0003
        self.noisy_observations = 2
        self.update = update
        self.regularization_param = regularization_param
        self.mirrordesc = LPMirrorDescent(12)

    def episode(self, env):
        # reset environment
        observation = self.transition(env, None)
        done = False
        history = []

        totalreward = 0.0
        while not done:
            s = observation
            a = self.choose_action(s)
            observation, reward, done, info = self.transition(env, a)
            totalreward += reward

            history.append((s, a, reward))

        for i in xrange(len(history)):
            s, a, reward = history[i]

            # Calculate G the total return
            totalreturn = reward
            decay = self.gamma
            for h in history[i+1:]:
                _, _, r = h
                totalreturn += decay*r
                decay *= self.gamma

            # SGD Update
            #olda = self.alpha
            #self.alpha = self.alpha / (1. + math.log(1.+i))
            self.weights = self.update(self, self.weights, self.gamma ** i, totalreturn, s, a)
            #self.alpha = olda

        return totalreward


    """
    Update methods
    """

    @staticmethod
    def sgd(reinforce, weights, gamma, totalreturn, s, a):
        return weights + reinforce.alpha*gamma*totalreturn*reinforce.policy.loggradient(s, a, weights)

    @staticmethod
    def subgradient(reinforce, weights, gamma, totalreturn, s, a):
        return weights + reinforce.alpha*(gamma*totalreturn*reinforce.policy.loggradient(s, a, weights) - reinforce.regularization_param*np.sign(weights))

    @staticmethod
    def proximal(reinforce, weights, gamma, totalreturn, s, a):
        # remember, lambda vars leak outside scope
        shrinkage = lambda x, alpha: (np.absolute(x) - alpha).clip(min=0.) * np.sign(x)

        y = weights + reinforce.alpha*gamma*totalreturn*reinforce.policy.loggradient(s, a, weights)
        return shrinkage(y, reinforce.alpha*reinforce.regularization_param)

    @staticmethod
    def mirrordescent(reinforce, weights, gamma, totalreturn, s, a):
        return reinforce.mirrordesc.update(weights, reinforce.alpha, -gamma*totalreturn*reinforce.policy.loggradient(s, a, weights))

    @staticmethod
    def mirrordescentprox(reinforce, weights, gamma, totalreturn, s, a):
        # remember, lambda vars leak outside scope
        shrinkage = lambda x, alpha: (np.absolute(x) - alpha).clip(min=0.) * np.sign(x)

        return reinforce.mirrordesc.update(weights, reinforce.alpha, -gamma*totalreturn*reinforce.policy.loggradient(s, a, weights), prox=lambda u: shrinkage(u, reinforce.alpha*reinforce.regularization_param))

#    @staticmethod
#    def frankwolfe(reinforce, weights, gamma, totalreturn, s, a):
#        gam = 120.0
#
#        grad = -gamma*totalreturn*reinforce.policy.loggradient(s, a, weights)
#
#        # We want argmin <s, grad f> s.t. ||s||_1 <= gam
#        # So, we can just find i = argmax |grad f|_i
#        # And set s_i = gam * sign((grad f)_i)
#        i = np.argmax(np.absolute(grad))
#        s_t = np.zeros(grad.shape[0])
#        s_t[i] = gam if grad[i] <= 0.0 else -gam
#
#        reinforce.fwdec += 1
#        return weights + (2. / (reinforce.fwdec)) * (s_t - weights)


    """
    Executes a step in the environment
    -- resets environment if action is none
    Returns next state, reward, done, info
    """
    def transition(self, env, action):
        # Add noise for teh lulz
        add_noise = lambda obs: np.append(obs, 0.3*np.random.rand(self.noisy_observations))

        if action is None:
            return add_noise(env.reset())
        else:
            obs, r, done, info = env.step(action)
            return (add_noise(obs), r, done, info)

    def choose_action(self, state):
        return np.random.choice(self.actions, p=[self.policy.policy(state, a, self.weights) for a in self.actions])

