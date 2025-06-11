import numpy as np

from src.adversaries.adversary import Adversary

# Return rewards based on calculated probability that agent will choose reward
class ProbabilisticAdversary(Adversary):
    def __init__(self, K,T):
        super().__init__()

        self.K = K
        self.probabilities = np.ones(K) / K
        self.T = T
        self.rewards = np.zeros(shape=(K, T))
        self.rewards[:, 0] = 1 - self.probabilities
        self.estimated_rewards = np.zeros(K)
        self.best_arm = 0
        self.eta = np.sqrt(np.log(K) / (T*K))

    # def get_reward(self, action: int, context: np.ndarray) -> float:
    #     t = len(self.history)
    #
    #     reward = self.rewards[action, t]
    #
    #     # print(f'action: {action}, reward: {reward}, t: {t}')
    #     # print(f'prob: {self.probabilities}')
    #     self.update_probabilities(action, reward, t)
    #     self.update_history(action, reward)
    #
    #     if t < self.T-1:
    #         sorted_indices = np.argsort(self.probabilities)
    #         cumulative_prob = 0.0
    #         reward_indices = []
    #
    #         for i in sorted_indices:
    #             prob = self.probabilities[i]
    #             if cumulative_prob + prob > self.lb:
    #                 break
    #             if cumulative_prob + prob <= self.ub:
    #                 reward_indices.append(i)
    #                 cumulative_prob += prob
    #             else:
    #                 break
    #
    #         self.rewards[:, t + 1] = 0
    #         self.rewards[reward_indices, t + 1] = 1
    #
    #     self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
    #
    #     return reward

    def get_best_reward(self, t):
        # t = len(self.history)
        return self.rewards[self.best_arm, t]


    def update_probabilities(self, action, reward, t):
        # Keep track of how exp3 would estimate the rewards
        P_ti = self.probabilities[action]
        delta = 1e-6
        estimated_reward = (1 - reward) / (P_ti + delta)

        # Update S_hat as per: Ŝ_ti = Ŝ_{t−1,i} + 1 - I{At=i} * (Xt / Pt[i])
        self.estimated_rewards += 1 - np.eye(self.K, dtype=int)[action] * estimated_reward
        # print(f'estimated rewards: {self.estimated_rewards}')

        weights = np.exp(self.estimated_rewards * self.eta)
        self.probabilities = weights / (np.sum(weights) + delta)
        # action_explorations = self.action_counts + np.maximum(1/(self.action_counts+1), 0.1)
        # means = np.divide(
        #     self.total_rewards,
        #     action_explorations,
        #     out=np.zeros_like(self.total_rewards),
        #     where=self.action_counts != 0
        # )
        # self.probabilities = means / sum(means)


    def reset(self):
        super().reset()
        self.probabilities = np.ones(self.K) / self.K
        self.rewards = np.zeros(shape=(self.K, self.T))
        self.rewards[:, 0] = 1 - self.probabilities
        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))

        self.estimated_rewards = np.zeros(self.K)

class TargetedProbabilityExploitationAdversary(ProbabilisticAdversary):
    def __init__(self, K,T, lb=1/4, ub=1/2):
        super().__init__(K,T)

        self.lb = lb
        self.ub = ub

    def get_reward(self, action: int, context: np.ndarray) -> float:
        t = len(self.history)
        reward = self.rewards[action, t]

        # print(f'action: {action}, reward: {reward}, t: {t}')
        # print(f'prob: {self.probabilities}')
        self.update_probabilities(action, reward, t)
        self.update_history(action, reward)


        if t < self.T-1:
            sorted_indices = np.argsort(self.probabilities)
            cumulative_prob = 0.0
            reward_indices = []

            for i in sorted_indices:
                prob = self.probabilities[i]
                if cumulative_prob + prob > self.lb:
                    break
                if cumulative_prob + prob <= self.ub:
                    reward_indices.append(i)
                    cumulative_prob += prob
                else:
                    break

            self.rewards[:, t + 1] = 0
            self.rewards[reward_indices, t + 1] = 1

        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))

        return reward



class InverseProbabilityRewardAdversary(ProbabilisticAdversary):
    def __init__(self, K,T, initial_reward_strategy=0):
        super().__init__(K,T)
        # initial reward strategy=0 is Random reward initialization
        # initial reward strategy=1 is reward damping
        self.initial_reward_strategy = initial_reward_strategy
        self.switch = np.floor(min(T/10, 5*K))

    def get_reward(self, action: int, context: np.ndarray) -> float:
        t = len(self.history)

        reward = self.rewards[action, t]

        # print(f'action: {action}, reward: {reward}, t: {t}')
        # print(f'prob: {self.probabilities}')
        # print(f'rewards: {self.rewards[:, t]}')
        self.update_probabilities(action, reward, t)
        self.update_history(action, reward)

        if t < self.switch:
            if self.initial_reward_strategy == 0:
                self.rewards[:, t + 1] = self.rewards[:, t]
            elif self.initial_reward_strategy == 1:
                self.rewards[:, t + 1] = np.minimum(self.rewards[:, t] + 0.02, 0.5)

        elif t < self.T-1:
            self.rewards[:, t + 1] = 1 - self.probabilities

            self.best_arm = np.argmax(np.sum(self.rewards, axis=1))

        return reward

    def reset(self):
        super().reset()
        if self.initial_reward_strategy == 0:
            self.rewards[:, 0] = np.random.uniform(0, 1, self.K)
        else:
            self.rewards[:, 0] = np.zeros(shape=self.K)

