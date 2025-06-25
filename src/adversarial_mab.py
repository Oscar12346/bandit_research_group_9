import numpy as np

from src.agents.agent import Agent
from src.environments.environment import Environment

class AdversarialMultiArmedBandit:
    def __init__(self):
        pass

    def play(self, environment: Environment, agent: Agent, num_sim: int, horizon: int, k: int):

        # init results
        rewards = np.zeros((num_sim, horizon))
        avg_rewards = np.zeros((num_sim, horizon))
        regrets = np.zeros((num_sim, horizon))
        pseudo_regrets = np.zeros((num_sim, horizon))
        cumulative_regrets = np.zeros((num_sim, horizon))
        cumulative_pseudo_regrets = np.zeros((num_sim, horizon))
        arm_rewards = np.zeros((num_sim, horizon, k))

        # Iterate over simulations
        for n in range(num_sim):
            agent.reset()

            # Iterate over time steps
            for t in range(horizon):
                action = agent.get_action()
                reward = environment.get_reward(action)
                agent.receive_reward(action, reward)

                # Get reward info
                means = environment.get_mean_rewards()
                best_reward = np.max(means)

                # Save data
                rewards[n,t] = reward
                avg_rewards[n,t] = means[action]
                regrets[n,t]= best_reward - reward
                pseudo_regrets[n,t] = best_reward - means[action]
                arm_rewards[n, t, :] = means

                # Compute cumulatives
                if t > 0: 
                    cumulative_regrets[n,t] = cumulative_regrets[n,t-1] + regrets[n,t]
                    cumulative_pseudo_regrets[n,t] = cumulative_pseudo_regrets[n,t-1] + pseudo_regrets[n,t]

                else: 
                    cumulative_regrets[n,t] = regrets[n,t]
                    cumulative_pseudo_regrets[n,t] = pseudo_regrets[n,t]

        fixed_policy_regret = np.zeros((num_sim, horizon))
        for n in range(num_sim):
            best_arm = np.argmax(np.sum(arm_rewards[n], axis=0))
            best_rewards = arm_rewards[n,:,best_arm]
            fixed_policy_regret[n] = np.cumsum(best_rewards - rewards[n])

        return rewards, regrets, avg_rewards, pseudo_regrets, cumulative_regrets, cumulative_pseudo_regrets, fixed_policy_regret