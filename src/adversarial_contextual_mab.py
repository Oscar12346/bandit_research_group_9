import numpy as np

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
        epsilons = np.zeros(num_sim)

        # Iterate over simulations
        for n in range(num_sim):
            contexts = []
            agent.reset()
            environment.reset() # resets the adversary
            actions = []

            # Iterate over time steps
            for t in range(horizon):
                # Interact with the environment
                context = environment.get_context()
                contexts.append(context)

                action = agent.get_action(context)
                actions.append(action)

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

            theta_sequence = environment.get_loss_vectors()  # (K, d)


            contexts = np.array(contexts)  # (T, d)
            theta_sequence = np.array(theta_sequence)  # (T, K, d)

            eps_max, eps_mean = self.estimate_epsilon(contexts, theta_sequence)
            epsilons[n] = eps_mean

            # loss_vectors = environment.get_loss_vectors() #returns self.theta
            # actual_losses = [contexts[t] @ loss_vectors[t][actions[t]] for t in range(horizon)]
            # Try fixed policies that always choose the same action
            # policy_losses = []
            # for a in range(len(loss_vectors[0])):
            #     loss = sum(contexts[t] @ loss_vectors[t][a] for t in range(horizon))
            #     policy_losses.append(loss)

            # best_fixed_policy_loss = min(policy_losses)
            # instantaneous_regret = np.zeros(horizon)
            #
            #
            # for t in range(horizon):
            # #     instantaneous_regret[t] = actual_losses[t] - min(
            # #         contexts[t] @ loss_vectors[t][a] for a in range(len(loss_vectors[0])))
            # # regrets[n] = np.cumsum(instantaneous_regret)
            #     context = contexts[t]
            #
            #     comparator_policy_action = np.argmax(context @ np.sum(loss_vectors, axis=0))
            #     comparator_reward = context @ loss_vectors[t, comparator_policy_action]
            #     expected_regret = comparator_reward - (rewards[n,t])
            #     if t > 0:
            #         expected_regrets[n,t] += expected_regrets[n, t-1] + expected_regret
            #     else:
            #         expected_regrets[n,t] = expected_regret
        return rewards, expected_regrets, avg_rewards, pseudo_regrets, cumulative_regrets, epsilons

    def estimate_epsilon(self,contexts, theta_seq):
        """
        Estimate epsilon for a non-linear adversary by computing how far the rewards deviate
        from a best linear approximation.

        :param contexts: List of context vectors x_t used during a simulation.
        :param adversary: The ContextualTargetedProbabilisticAdversary instance.
        :return: Estimated epsilon (scalar).
        """
        T, K, d = theta_seq.shape
        X = np.array(contexts)  # shape (T, d)

        epsilons = []

        for a in range(K):
            # Estimate best-fit fixed theta for this arm (OLS solution)
            A = theta_seq[:, a, :]  # shape (T, d)
            y = np.einsum("td,td->t", A, X)  # True rewards at each t: r_t = <theta_t,a, x_t>

            # Solve for best-fit linear approximation: theta_bar_a minimizing ||A_t x_t - <theta_bar, x_t>||
            # Solve least squares: min_theta ||X @ theta - y||^2
            theta_bar, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

            # Compute error at each time step
            y_hat = X @ theta_bar
            epsilon_a = np.abs(y - y_hat)
            epsilons.append(epsilon_a)

        epsilons = np.stack(epsilons, axis=1)  # shape (T, K)
        epsilon_max = np.max(epsilons)
        epsilon_mean = np.mean(epsilons)

        return epsilon_max, epsilon_mean
    # Alternative way to calculate regret which I want to run by Guyon
    # def play(self, agent: Agent, environment: AdversarialContextualEnv, num_sim: int, horizon: int) -> tuple:
    #     rewards = np.zeros((num_sim, horizon))
    #     cumulative_regrets = np.zeros((num_sim, horizon))
    #     policy_regrets = np.zeros((num_sim, horizon))
    #
    #     for n in range(num_sim):
    #         contexts = []
    #         actions = []
    #         agent.reset()
    #         environment.reset()
    #
    #         for t in range(horizon):
    #             context = environment.get_context()
    #             action = agent.get_action(context)
    #             reward = environment.get_reward(action, context)
    #             agent.receive_reward(action, reward)
    #
    #             contexts.append(context)
    #             actions.append(action)
    #             rewards[n, t] = reward
    #
    #         # Collect true loss vectors from adversary
    #         theta = environment.get_loss_vectors()  # shape: (T, K, d)
    #
    #         # Compute actual losses per step: context @ theta[action]
    #         actual_rewards = np.array([
    #             contexts[t] @ theta[t][actions[t]] for t in range(horizon)
    #         ])
    #
    #         # # Compute best fixed action in hindsight (policy comparator)
    #         K = theta[0].shape[0]
    #         # fixed_action_losses = [
    #         #     sum(contexts[t] @ theta[t][a] for t in range(horizon))
    #         #     for a in range(K)
    #         # ]
    #         # best_fixed_loss = min(fixed_action_losses)
    #         #
    #         # # Compute instantaneous and cumulative regret
    #         # for t in range(horizon):
    #         #     best_reward = max(contexts[t] @ theta[t][a] for a in range(K))
    #         #     policy_regrets[n, t] = best_reward - actual_rewards[t]
    #         #
    #         # cumulative_regrets[n] = np.cumsum(policy_regrets[n])
    #         # Step 1: compute total reward for each fixed action
    #         total_rewards_per_action = np.zeros(K)
    #         for a in range(K):
    #             total_rewards_per_action[a] = sum(contexts[t] @ theta[t][a] for t in range(horizon))
    #         print(total_rewards_per_action)
    #         # Step 2: find best fixed action
    #         best_fixed_action = np.argmax(total_rewards_per_action)
    #
    #         # Step 3: compute regret at each time step
    #         for t in range(horizon):
    #             best_fixed_reward = contexts[t] @ theta[t][best_fixed_action]
    #             policy_regrets[n, t] = best_fixed_reward - actual_rewards[t]
    #
    #         # Step 4: cumulative regret
    #         cumulative_regrets[n] = np.cumsum(policy_regrets[n])
    #         print(f'policy_regrets[n]={policy_regrets[n]}')
    #     return rewards, policy_regrets, 0, 0, cumulative_regrets, 0
