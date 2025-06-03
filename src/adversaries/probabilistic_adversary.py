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
        self.total_rewards = np.zeros(K)
        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
        self.action_counts = np.zeros(K)

    def get_reward(self, action: int, context: np.ndarray) -> float:
        t = len(self.history)

        reward = self.rewards[action, t]
        self.total_rewards[action] += reward
        self.action_counts[action] += 1
        print(f'action: {action}, reward: {reward}, t: {t}')
        print(f'prob: {self.probabilities}')
        self.update_probabilities(action, reward, t)
        self.update_history(action, reward)


        if t < self.T-1:
            self.rewards[:, t+1] = 1 - self.probabilities

        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
        return reward

    def get_best_reward(self, t: int):
        return self.rewards[self.best_arm, t]


    def update_probabilities(self, action, reward, t):
        action_explorations = self.action_counts + np.maximum(1/(self.action_counts+1), 0.1)
        means = np.divide(
            self.total_rewards,
            action_explorations,
            out=np.zeros_like(self.total_rewards),
            where=self.action_counts != 0
        )
        self.probabilities = means / sum(means)


    def reset(self):
        super().reset()
        self.probabilities = np.ones(self.K) / self.K
        self.rewards = np.zeros(shape=(self.K, self.T))
        self.rewards[:, 0] = 1 - self.probabilities / self.K
        self.total_rewards = np.zeros(self.K)
        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
        self.action_counts = np.zeros(self.K)


