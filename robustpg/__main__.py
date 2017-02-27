from __future__ import division
import gym
import functools
import sys
import os
import math
import numpy as np
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from features import FourierBasis
from policy import SoftmaxPolicy
from policy import GaussianPolicy
from pg import REINFORCE
from mirrordescent import LPMirrorDescent

class Features(object):

    """
    Creates separate (independent) weights per action
    """
    @staticmethod
    def simple(num_states, num_actions, s, a):
        feats = np.zeros(num_states * num_actions)
        feats[(num_states*a):(num_states*a + num_states)] = s
        return feats


"""
Run REINFORCE with different params
"""
def run_experiment(output_name, policy, env, init_weights, params_list, learning_rates, trials=10, steps=300):
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_name)
    except OSError:
        if not os.path.isdir(output_name):
            raise

    # Run experiments
    for p in params_list:
        pg = REINFORCE(policy, init_weights(), p, learning_rates)

        rewards = np.zeros((trials, steps))
        weights = np.zeros((trials, init_weights().shape[0]))
        for t in xrange(trials):
            for i in xrange(steps):
                rewards[t, i] = pg.episode(env)

            weights[t:] = pg.weights
            pg.weights = init_weights()

        # rewards standard deviation
        #rewards_std = np.std(rewards, axis=0)
        rewards_mean = np.mean(rewards, axis=0)

        # 95% Confidence intervals
        #data_crit = stats.t.ppf(1.0 - (0.05 / 2.0), trials - 1)
        #confidences = map(lambda s: (data_crit*s) / math.sqrt(trials), rewards_std)

        weights_mean = np.mean(weights, axis=0)

        # Save data
        np.save('%s/rewards.npy' % output_name, rewards, allow_pickle=False)
        np.save('%s/weights.npy' % output_name, weights, allow_pickle=False)

        plt.plot(range(len(rewards_mean)), rewards_mean, label=p['name'])
        #plt.errorbar(x=range(len(rewards_mean)), y=rewards_mean, label=p['name'], yerr=confidences, errorevery=10)

    plt.legend(loc=0)
    plt.axes().set_xlabel("Episodes")
    plt.axes().set_ylabel("Reward")
    plt.savefig('%s/%s.pdf' % (output_name, output_name))
    plt.show()


"""
Run REINFORCE on Condor with different params
This runs one trial (Use a trial per condor job)
"""
def run_condor_experiment(output_dir, trialnum, policy, env, init_weights, params, learning_rates, steps=300):
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir)
    except OSError:
        if not os.path.isdir(output_dir):
            raise

    # Run experiments
    pg = REINFORCE(policy, init_weights(), params, learning_rates)

    rewards = np.zeros(steps)
    weights = np.zeros(init_weights().shape[0])
    for i in xrange(steps):
        rewards[i] = pg.episode(env)

    # Save data
    np.savetxt('%s/rewards-%s.csv' % (output_dir, trialnum), rewards, delimiter=",")


def cartpole(args=None):
    # Set up environment and learner
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    ranges = zip(env.observation_space.low, env.observation_space.high)
    actions = range(env.action_space.n)

    noisy_observations = 2

    # State features
    features = functools.partial(Features.simple, state_size + noisy_observations, env.action_space.n)
    policy = SoftmaxPolicy(actions, features)

    # Initial weights
    init_weights = lambda: np.zeros((state_size + noisy_observations) * env.action_space.n)

    # Experiment parameters
    params = {
        'noisy_observations': noisy_observations,
        'regularization': 0.0,
        'mirror': None,
        'update': REINFORCE.sgd,
        'name': 'sgd'
    }

    ###############################################
    # Compare mpg, sparse pg, and normal REINFORCE
    ###############################################
    mpg = params.copy()
    mpg['mirror'] = LPMirrorDescent(12)
    mpg['update'] = REINFORCE.mirrordescent
    mpg['name'] = 'mirrored pg'

    spg = params.copy()
    spg['update'] = REINFORCE.proximal
    spg['regularization'] = 3.0
    spg['name'] = 'sparse pg'

    baseline = params.copy()
    baseline['name'] = 'baseline'

    params_list = [mpg, spg, baseline]

    run_experiment('allcompare', policy, env, init_weights, params_list, trials=200)

    """
    ###############################################
    # Compare mpg with various p values
    ###############################################
    params_list = []
    for pval in [2., 5., 12., 20., 30.]:
        mpg = params.copy()
        mpg['mirror'] = LPMirrorDescent(pval)
        mpg['update'] = REINFORCE.mirrordescent
        mpg['name'] = '$p = %.1f$' % pval
        params_list.append(mpg)


    run_experiment('mpgpcompare', policy, actions, env, init_weights, params_list, trials=200)
    """

    """
    ###############################################
    # Compare sparse pg with various regularization parameters
    ###############################################
    params_list = []
    for lam in [0., 2., 3., 4., 5.]:
        spg = params.copy()
        spg['regularization'] = lam
        spg['update'] = REINFORCE.proximal
        spg['name'] = '$\lambda = %.1f$' % lam
        params_list.append(spg)


    run_experiment('spglcompare', policy, actions, env, init_weights, params_list, trials=200)
    """


def pendulum(args):
    if len(args) != 3:
        print("Check CLI Arguments")
        sys.exit(1)
    trialnum, outputdir, numepisodes = args
    numepisodes = int(numepisodes)

    # Set up environment and learner
    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape[0]
    ranges = zip(env.observation_space.low, env.observation_space.high)
    action_ranges = zip(env.action_space.low, env.action_space.high)

    # Features
    feats = FourierBasis(state_size, 3)
    def phi(st):
        state = np.copy(st)
        state[0] = (state[0] + 1.) / 2.
        state[1] = (state[1] + 1.) / 2.
        state[2] = (state[2] + 8.) / 16.
        return feats.generate_features(state)

    # State features
    policy = GaussianPolicy(phi)

    # Initial weights
    init_weights = lambda: np.zeros(feats.num_basis_functions() * 2)

    # Experiment parameters
    params = {
        'noisy_observations': 0,
        'regularization': 0.0,
        'mirror': None,
        'update': REINFORCE.sgd,
        'name': 'sgd'
    }

    ###############################################
    # Compare mpg, sparse pg, and normal REINFORCE
    ###############################################
    #mpg = params.copy()
    #mpg['mirror'] = LPMirrorDescent(12)
    #mpg['update'] = REINFORCE.mirrordescent
    #mpg['name'] = 'mirrored pg'

    #spg = params.copy()
    #spg['update'] = REINFORCE.proximal
    #spg['regularization'] = 3.0
    #spg['name'] = 'sparse pg'

    baseline = params.copy()
    baseline['name'] = 'baseline'

    #params_list = [mpg, spg, baseline]
    params = baseline

    run_condor_experiment(outputdir, trialnum, policy, env, init_weights, params, lambda l: np.tile(feats.learning_rates(l), 2), steps=numepisodes)


def main(args):
    pendulum(args)


if __name__ == "__main__":
    main(sys.argv[1:])
