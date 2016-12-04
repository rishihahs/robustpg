from __future__ import division
import gym
import functools
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from policy import SoftmaxPolicy
from pg import REINFORCE

class Features(object):

    """
    Creates separate (independent) weights per action
    """
    @staticmethod
    def simple(num_states, num_actions, s, a):
        feats = np.zeros(num_states * num_actions)
        feats[(num_states*a):(num_states*a + num_states)] = s
        return feats


def main(args=None):
    # Set up environment and learner
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    ranges = zip(env.observation_space.low, env.observation_space.high)
    actions = range(env.action_space.n)

    noisy_observations = 2

    # State features
    features = functools.partial(Features.simple, state_size + noisy_observations, env.action_space.n)
    policy = SoftmaxPolicy(actions, features)


    # Compare mpg, sparse pg, and normal REINFORCE
    for update_method, update_name in [(REINFORCE.mirrordescent, 'mpg'), (REINFORCE.proximal, 'prox-pg'), (REINFORCE.sgd, 'sgd')]:
        pg = REINFORCE(policy, actions, np.zeros((state_size + noisy_observations) * env.action_space.n), update_method, regularization_param=1.0)

        trials = 10
        steps = 300

        rst = np.zeros((trials, steps))
        whist = np.zeros((trials, (state_size + noisy_observations) * env.action_space.n))
        for t in xrange(trials):
            for i in xrange(steps):
                rst[t, i] = pg.episode(env)

            whist[t:] = pg.weights
            pg.weights = np.zeros((state_size + noisy_observations) * env.action_space.n)

        rs = np.sum(rst, axis=0) / trials
        ws = np.sum(whist, axis=0) / trials
        print(ws)

        plt.plot(range(len(rs)), rs, label=update_name)


    plt.legend(loc=0)
    plt.axes().set_xlabel("Episodes")
    plt.axes().set_ylabel("Reward")
    plt.savefig('allcompare.pdf')
    plt.show()


if __name__ == "__main__":
    main()
