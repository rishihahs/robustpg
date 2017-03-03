from __future__ import division
import random
import math
import numpy as np

class REINFORCE(object):

    def __init__(self, policy, initial_weights, params, learning_rates=lambda l: l):
        self.policy = policy
        self.weights = initial_weights
        self.gamma = 1.
        self.alpha = 0.0000001
        self.learning_rates = learning_rates(self.alpha)
        self.noisy_observations = params.get('noisy_observations', 2)
        self.update = params.get('update', None)
        self.regularization_param = params.get('regularization', 0.)
        self.mirrordesc = params.get('mirror', None)

    def episode(self, env):
        # reset environment
        observation = self.transition(env, None)
        done = False
        history = []

        totalreward = 0.0
        steps = 0
        while not done:
            s = observation
            a = self.choose_action(s)
            observation, reward, done, info = self.transition(env, a)
            totalreward += reward

            history.append((s, a, reward))

            #print(s)
            #print(a)
            #print(reward)
            #print(observation)
            #print("\n")
            if math.isnan(reward):
                print("NAN!!!")
                import sys; sys.exit()
            steps += 1
            if steps >= 200:
                break

        #print("DONE--")

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
            olda = self.alpha
            self.alpha = self.alpha / (1. + i)
            #print(self.weights)
            self.weights = self.update(self, self.weights, self.gamma ** i, totalreturn, s, a)
            #print(self.weights)
            self.alpha = olda

        return totalreward


    """
    Update methods
    """

    @staticmethod
    def sgd(reinforce, weights, gamma, totalreturn, s, a):
        return weights + reinforce.learning_rates*gamma*totalreturn*reinforce.policy.loggradient(s, a, weights)

    # TODO: Use learning_rates instead of alpha
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
        return self.policy.sample(state, self.weights)


class ActorCritic(object):

    def __init__(self, actor, critic, actor_weights, critic_weights, params,
                    actor_learning_rates=lambda l1: l1, critic_learning_rates=lambda l2: l2):
        self.actor = actor
        self.critic = critic
        self.actor_weights = actor_weights
        self.critic_weights = critic_weights

        self.gamma = 1.0
        self.base_actor_learning_rate = 0.00001
        self.base_critic_learning_rate = 0.00001
        self.actor_lambda = 0.7
        self.critic_lambda = 0.7

        self.actor_learning_rate = actor_learning_rates(self.base_actor_learning_rate)
        self.critic_learning_rate = critic_learning_rates(self.base_critic_learning_rate)

        self.noisy_observations = params.get('noisy_observations', 2)
        self.update = params.get('update', None)
        self.regularization_param = params.get('regularization', 0.)
        self.mirrordesc = params.get('mirror', None)

    def episode(self, env):
        # reset environment
        observation = self.transition(env, None)
        done = False

        # eligibility traces
        e_actor = np.zeros(self.actor_weights.shape)
        e_critic = np.zeros(self.critic_weights.shape)

        # gamma decay
        I = 1.

        totalreward = 0.0
        steps = 0
        while not done:
            s = observation
            a = self.choose_action(s)
            observation, reward, done, info = self.transition(env, a)

            if done:
                reward += 2600
                #print("----- Woohooo -------")

            # Reward Shaping
            #pi_s = abs(s[1]) / 0.07 + (s[0] + 1.2) / 1.8
            #pi_sprime = abs(observation[1]) / 0.07 + (observation[0] + 1.2) / 1.8
            #shaping = self.gamma*pi_sprime - pi_s
            #reward += shaping

            totalreward += reward

            steps += 1
            if steps >= 2000:
                #print("ep")
                done = True

            delta = reward + self.gamma*self.critic.value(observation, self.critic_weights)
            delta -= 0 if done else self.critic.value(s, self.critic_weights)

            e_actor = self.actor_lambda*e_actor + I*self.actor.loggradient(s, a, self.actor_weights)
            e_critic = self.critic_lambda*e_critic + I*self.critic.gradient(s, self.critic_weights)

            self.actor_weights = self.actor_weights + self.actor_learning_rate*delta*e_actor
            self.critic_weights = self.critic_weights + self.critic_learning_rate*delta*e_critic

            I = self.gamma*I

            if math.isnan(reward):
                print("NAN!!!")
                import sys; sys.exit()

        return totalreward


    """
    Update methods
    """
    @staticmethod
    def sgd(reinforce, weights, gamma, totalreturn, s, a):
        return weights + reinforce.learning_rates*gamma*totalreturn*reinforce.policy.loggradient(s, a, weights)


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
        return self.actor.sample(state, self.actor_weights)

