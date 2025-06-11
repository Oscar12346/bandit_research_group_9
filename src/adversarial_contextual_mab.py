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

        # Iterate over simulations
        for n in range(num_sim):
            agent.reset()
            environment.reset() # resets the adversary
            # Iterate over time steps
            for t in range(horizon):
                # Interact with the environment
                context = environment.get_context()
                action = agent.get_action(context)
                reward = environment.get_reward(action, context)
                agent.receive_reward(action,reward)

                # Get reward info
                means = environment.get_mean_rewards(context)
                best_reward = np.max(means)

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

        return rewards, regrets, avg_rewards, pseudo_regrets, cumulative_regrets, cumulative_pseudo_regrets