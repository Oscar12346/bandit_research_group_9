import numpy as np
from numpy import ndarray

from src.adversaries.adversary import Adversary


class DeceptiveAdversary(Adversary):
    def __init__(self, K):
        super().__init__()

        self.history = []
        self.K = K
        self.best_arm = np.random.choice(K)
        self.switch_arm = np.random.choice(np.setdiff1d(range(self.K), self.best_arm))

    def update_history(self, action, reward):
        self.history.append((action, reward))

    def get_reward(self, action: int, context: ndarray) -> float:
        t = len(self.history)

        reward = self.get_arm_reward(action, context, t)
        self.update_history(action, reward)
        return reward

    def get_arm_reward(self, action: int, context: ndarray, t: int) -> float:
        if t < 25:
            return 0.9 if action == self.best_arm else 0.25
        else:
            return 0.5 if action == self.switch_arm else 0.1

    def get_best_reward(self, t: int):
        return self.get_arm_reward(self.switch_arm, None, t)

    def reset(self):
        self.history = []