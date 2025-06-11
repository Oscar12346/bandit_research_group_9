import numpy as np

from src.adversaries.deceptive_adversary import DeceptiveAdversary
from src.adversaries.probabilistic_adversary import ProbabilisticAdversary, TargetedProbabilityExploitationAdversary, InverseProbabilityRewardAdversary
from src.agents.exp3 import Exp3
from src.agents.old_agents import EpsilonGreedy
from src.display import display
from src.environments.environment import *
from src.environments.adversarial_mab_env import *


class AdversarialMultiArmedBandit:
    def __init__(self):
        pass

    def play_mab(self, environment, agent, N, T):
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
        adversary = environment.get_adversary()

        for n in range(N):
            agent.reset()
            adversary.reset()
            for t in range(T):
                action = agent.get_action()
                reward = environment.get_reward(action)
                agent.receive_reward(action, reward)

                # compute instantaneous reward  and (pseudo) regret
                rewards[n, t] = reward

                # means = environment.get_means()
            for t in range(T):
                best_reward = adversary.get_best_reward(t)
                regrets[
                    n, t] = best_reward - rewards[n, t]  # this can be negative due to the noise, but on average it's positive
                avg_rewards[n, t] = 0
                pseudo_regrets[n, t] = 0

        return agent.name(), rewards, regrets, avg_rewards, pseudo_regrets

    def experiment_mab(self, environment, agents, N, T, mode="regret"):
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
            agent_id, rewards, regrets, avg_rewards, pseudo_regrets = self.play_mab(environment, agent, N, T)

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

#  Runnable for Multi armed bandit environments (this includes adversarial ones)
if __name__ == "__main__":

    K = 5  # number of arms
    T = 10000  # Horizon
    N = 100  # number of simulations
    # Example of normal MAB env initialization
    strats = [0,1]
    experiments = {}
    # Visualization
    Nsub = 100  # Subsampled points
    tsav = range(2, T, Nsub)
    lr = np.sqrt(np.log(K) / (T * K))
    amab = AdversarialMultiArmedBandit()
    bounds_per_timestep = np.array([2 * np.sqrt(t * K * np.log(K)) for t in range(T)])
    exp3 = Exp3(K, lr=lr)
    for strat in strats:
        adversary = InverseProbabilityRewardAdversary(K, T, initial_reward_strategy=strat)
        random_mab_env = Adversarial_MAB_env(K, adversary)

        experiment = amab.experiment_mab(random_mab_env, [exp3], N=N, T=T, mode="regret")

        # Modify the label to include K and strategy
        for agent_name, data in experiment.items():
            label = f"{agent_name}_K={K}_strat={strat}"
            experiments[label] = data

    display.plot_result(experiments, q=10, mode="regret", cumulative=True,
                        bounds_per_timestep=bounds_per_timestep)




    # eps_greedy = EpsilonGreedy(K, 0.01)
    #
    # experiment = amab.experiment_mab(random_mab_env, [exp3], N=N, T=T, mode="regret")
    #
    # display.plot_result(experiment, q=10, mode="regret", cumulative=True, bounds_per_timestep=bounds_per_timestep)

    # greedy = bandit_solutions.EpsilonGreedy(K, eps=0.1)
    #
    # experiment2 = experiment_mab(env, [greedy], N=N, T=T, mode="reward")
    # display.plot_result(experiment2, q=10, mode="reward", cumulative=False)