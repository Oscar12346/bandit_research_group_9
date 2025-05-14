# Some standard imports
import numpy as np
from scipy.stats import bernoulli
from math import log
import random
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from bandit_practice.model import Random_Adversarial_MAB_env

matplotlib.style.use('seaborn-v0_8')

import model
from model import Environment, Agent, MAB_env
import display
from bandit_solutions import Exp3
import bandit_solutions

def play_mab(environment, agent, N, T):
    """
    Play N independent runs of length T for the specified agent.

    :param environment: a MAB instance
    :param agent: a bandit algorithm
    :param N: number of independent simulations
    :param T: decision horizon
    :return: the agent's name, and the collected data in numpy arrays
    """

    rewards = np.zeros((N, T))
    regrets = np.zeros((N, T))
    pseudo_regrets = np.zeros((N, T))
    avg_rewards = np.zeros((N, T))

    for n in range(N):
        agent.reset()
        for t in range(T):
            action = agent.get_action()
            reward = environment.get_reward(action)
            agent.receive_reward(action,reward)

            # compute instantaneous reward  and (pseudo) regret
            rewards[n,t] = reward
            means = environment.get_means()
            best_reward = np.max(means)
            regrets[n,t]= best_reward - reward # this can be negative due to the noise, but on average it's positive
            avg_rewards[n,t] = means[action]
            pseudo_regrets[n,t] = best_reward - means[action]

    return agent.name(), rewards, regrets, avg_rewards, pseudo_regrets


def experiment_mab(environment, agents, N, T, mode="regret"):
    """
    Play N trajectories for all agents over a horizon T. Store data in a dictionary.

    :param environment: a MAB instance
    :param agent: a list of bandit algorithms to compare
    :param N: number of independent simulations
    :param T: decision horizon
    :param mode: the performance measure to return ("reward", "average reward", "regret", "pseudo regret")
    :return: the performance for each agent in a dictionary indexed by the agent's name
    """

    all_data = {}

    for agent in agents:
        agent_id, rewards, regrets, avg_rewards, pseudo_regrets = play_mab(environment, agent, N, T)

        if mode == "regret":
            all_data[agent_id] = regrets
        elif mode == "pseudo regret":
            all_data[agent_id] = pseudo_regrets
        elif mode == "reward":
            all_data[agent_id] = rewards
        elif mode == "average reward":
            all_data[agent_id] = avg_rewards
        else:
            raise ValueError

    return all_data




if __name__ == "__main__":
    K = 3  # number of arms

    # We should change this to our adversarial strategy for rewards
    # This is now random rewards at each step.
    env = Random_Adversarial_MAB_env(10)  # We don't know the reward distributions in advance!

    T = 1000  # Horizon
    N = 50  # number of simulations

    # Visualization
    Nsub = 100  # Subsampled points
    tsav = range(2, T, Nsub)

    exp3 = Exp3(K)
    experiment = experiment_mab(env, [exp3], N=N, T=T, mode="regret")

    display.plot_result(experiment, q=10, mode="regret", cumulative=False)

    # greedy = bandit_solutions.EpsilonGreedy(K, eps=0.1)
    #
    # experiment2 = experiment_mab(env, [greedy], N=N, T=T, mode="reward")
    # display.plot_result(experiment2, q=10, mode="reward", cumulative=False)


