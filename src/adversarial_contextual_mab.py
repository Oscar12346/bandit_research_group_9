import numpy as np
import vis

from src.adversaries.contextual_probabilistic_adversary import ContextualProbabilisticAdversary
from src.agents.agent import Agent
from src.agents.robust_lin_exp3 import RobustLinExp3Agent
from src.contexts.gaussian_context import GaussianContext
from src.environments.adversarial_contextual_env import AdversarialContextualEnv

class AdversarialContextualMAB:

    def __init__(self):
        pass 

    def play(self, agent: Agent, environment: AdversarialContextualEnv, num_sim: int, horizon: int) -> tuple:
        
        # init results
        rewards = np.zeros((num_sim, horizon))
        avg_rewards = np.zeros((num_sim, horizon))
        regrets = np.zeros((num_sim, horizon))
        pseudo_regrets = np.zeros((num_sim, horizon))
        cumulative_regrets = np.zeros((num_sim, horizon))
        cumulative_pseudo_regrets = np.zeros((num_sim, horizon))
        expected_regrets = np.zeros((num_sim, horizon))

        # Iterate over simulations
        for n in range(num_sim):
            contexts = []
            agent.reset()
            environment.reset() # resets the adversary

            # Iterate over time steps
            for t in range(horizon):
                # Interact with the environment
                context = environment.get_context()
                contexts.append(context)

                action = agent.get_action(context)

                reward = environment.get_reward(action, context)
                agent.receive_reward(action,reward)

                # Get reward info
                means = environment.get_mean_rewards(context)
                best_reward = np.max(means)
                if reward > best_reward + 1e6:
                    print(f'best_reward: {best_reward} actual reward: {reward}')

                # Save data
                rewards[n,t] = reward
                # avg_rewards[n,t] = means[action]
                regrets[n,t]= best_reward - reward
                # pseudo_regrets[n,t] = best_reward - means[action]

                # Compute cumulatives
                if t > 0:
                    cumulative_regrets[n,t] = cumulative_regrets[n,t-1] + regrets[n,t]
                    # cumulative_pseudo_regrets[n,t] = cumulative_pseudo_regrets[n,t-1] + pseudo_regrets[n,t]
                else:
                    cumulative_regrets[n,t] = regrets[n,t]
                    # cumulative_pseudo_regrets[n,t] = pseudo_regrets[n,t]
            # loss_vectors = environment.get_loss_vectors() #returns self.theta
            # for t in range(horizon):
            #     context = contexts[t]
            #     comparator_policy_action = np.argmin(context @ np.sum(loss_vectors, axis=0))
            #     comparator_loss = context @ loss_vectors[t, comparator_policy_action]
            #     expected_regret = comparator_loss - (-rewards[n,t])
            #     if t > 0:
            #         expected_regrets[n,t] += expected_regrets[n, t-1] + expected_regret
            #     else:
            #         expected_regrets[n,t] = expected_regret
        return rewards, regrets, avg_rewards, pseudo_regrets, cumulative_regrets, cumulative_pseudo_regrets