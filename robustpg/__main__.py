from __future__ import division
import gym
import functools
import os
import math
import numpy as np
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from features import FourierBasis
from policy import SoftmaxPolicy
from policy import GaussianPolicy
from pg import REINFORCE
from pg import ActorCritic
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


class LinearVFA(object):

    def __init__(self, featuregen):
        self.featuregen = featuregen

    def value(self, state, weights):
        return self.featuregen.linear_combination(weights, state)

    def gradient(self, state, weights):
        return self.featuregen.generate_features(state)


"""
Vizualize VFA
observation_indices: tuple of the indices of the two state features desired
features_gen: function that takes two state features and generates full features
action: action to visualize
"""
def visualize_vfa(vfa, env, weights):
    maxr = 600.0  # Maximum range
    minr = -600.0 # Minimum range

    i1, i2 = (0, 1)

    x_range = [max(env.observation_space.low[i1], minr), min(env.observation_space.high[i1], maxr)]
    y_range = [max(env.observation_space.low[i2], minr), min(env.observation_space.high[i2], maxr)]

    # This determines how densely the grid is. To cut down on computational
    # cost, we'll keep it a bit sparse.
    x_resolution = (x_range[1] - x_range[0])/50.0
    y_resolution = (y_range[1] - y_range[0])/50.0

    x = np.arange(x_range[0], x_range[1], x_resolution)
    y = np.arange(y_range[0], y_range[1], y_resolution)
    X, Y = np.meshgrid(x, y)
    zs = np.array([vfa.value(np.array([x, y]), weights) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    plt.cla()
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value function')
    plt.draw()
    plt.show()


"""
Run REINFORCE with different params
"""
def run_experiment(output_name, policy, critic, env, init_actor_weights, init_critic_weights, params_list, actor_learning_rates, critic_learning_rates, trials=10, steps=300):
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_name)
    except OSError:
        if not os.path.isdir(output_name):
            raise

    # Run experiments
    for p in params_list:
        #pg = REINFORCE(policy, init_weights(), p, learning_rates)
        pg = ActorCritic(policy, critic, init_actor_weights(), init_critic_weights(), p, actor_learning_rates, critic_learning_rates)

        rewards = np.zeros((trials, steps))
        actor_weights = np.zeros((trials, init_actor_weights().shape[0]))
        critic_weights = np.zeros((trials, init_critic_weights().shape[0]))
        for t in xrange(trials):
            for i in xrange(steps):
                rewards[t, i] = pg.episode(env)

                #if i > 0 and i % 20 == 0:
                #    visualize_vfa(critic, env, pg.critic_weights)

            actor_weights[t:] = pg.actor_weights
            pg.actor_weights = init_actor_weights()
            pg.critic_weights = init_critic_weights()

        # rewards standard deviation
        #rewards_std = np.std(rewards, axis=0)
        rewards_mean = np.mean(rewards, axis=0)

        # 95% Confidence intervals
        #data_crit = stats.t.ppf(1.0 - (0.05 / 2.0), trials - 1)
        #confidences = map(lambda s: (data_crit*s) / math.sqrt(trials), rewards_std)

        weights_mean = np.mean(actor_weights, axis=0)

        # Save data
        np.save('%s/rewards.npy' % output_name, rewards, allow_pickle=False)
        np.save('%s/weights.npy' % output_name, actor_weights, allow_pickle=False)

        plt.plot(range(len(rewards_mean)), rewards_mean, label=p['name'])
        #plt.errorbar(x=range(len(rewards_mean)), y=rewards_mean, label=p['name'], yerr=confidences, errorevery=10)

    plt.legend(loc=0)
    plt.axes().set_xlabel("Episodes")
    plt.axes().set_ylabel("Reward")
    plt.savefig('%s/%s.pdf' % (output_name, output_name))
    plt.show()


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


def pendulum(args=None):
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

    # Critic
    critic = LinearVFA(feats)

    # State features
    policy = GaussianPolicy(phi)

    # Initial weights
    init_actor_weights = lambda: np.zeros(feats.num_basis_functions() * 2)
    init_critic_weights = lambda: np.zeros(feats.num_basis_functions())

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
    params_list = [baseline]

    run_experiment('test3', policy, critic, env, init_actor_weights, init_critic_weights, params_list, lambda l: np.tile(feats.learning_rates(l), 2), feats.learning_rates, trials=1, steps=2500)


def mountaincar(args=None):
    #if len(args) != 3:
    #    print("Check CLI Arguments")
    #    sys.exit(1)
    #trialnum, numepisodes, outputdir = args
    #numepisodes = int(numepisodes)

    # Set up environment and learner
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    ranges = zip(env.observation_space.low, env.observation_space.high)
    actions = range(env.action_space.n)

    # Features
    criticfeats = FourierBasis(state_size, 7)
    feats = FourierBasis(state_size+1, 3)
    #def phi(st, a):
    #    state = np.copy(st)
    #    state[0] = (state[0] + 1.2) / 1.8
    #    state[1] = (state[1] + 0.07) / 0.14
    #
    #    f = np.zeros(state_size * env.action_space.n)
    #    f[(state_size*a):(state_size*a + state_size)] = state
    #
    #    return feats.generate_features(f)
    def phi(st, a):
        state = np.copy(st)
        state[0] = (state[0] + 1.2) / 1.8
        state[1] = (state[1] + 0.07) / 0.14

        f = np.zeros(state_size + 1)
        f[:state_size] = state
        f[-1] = a / 2.

        return feats.generate_features(f)

    # Critic
    critic = LinearVFA(criticfeats)

    # State features
    policy = SoftmaxPolicy(actions, phi)

    # Initial weights
    init_actor_weights = lambda: np.zeros(feats.num_basis_functions())
    init_critic_weights = lambda: np.zeros(criticfeats.num_basis_functions())

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
    #params = baseline
    params_list = [baseline]

    #run_condor_experiment(outputdir, trialnum, policy, env, init_weights, params, feats.learning_rates, steps=numepisodes)
    run_experiment('test3', policy, critic, env, init_actor_weights, init_critic_weights, params_list, feats.learning_rates, criticfeats.learning_rates, trials=1, steps=100)



def main(args=None):
    mountaincar()


if __name__ == "__main__":
    main()
