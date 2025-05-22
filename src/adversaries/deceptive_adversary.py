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
        print(f'best arm and switch arm {self.best_arm, self.switch_arm}')

    def update_history(self, action, reward):
        self.history.append((action, reward))

    def get_reward(self, action: int, context: ndarray) -> float:
        t = len(self.history)

        # Implementation now expects that there are 2 actions,
        # when we extend this Adversary to have knowledge about the amount of actions we can extend
        if t < 25:
            # Make it seem that action 0 is really good and the rest worse in the first 500 timesteps
            reward = 0.9 if action == self.best_arm else 0.4
            self.update_history(action, reward)
            return reward
        else:
            # Now we expose which action is actually the best, which is action 1
            # All other actions receive no reward anymore
            reward = 0.9 if action == self.switch_arm else 0.1
            self.update_history(action, reward)
            return reward


    def get_best_reward(self):
        return 1.0

    def reset(self):
        self.history = []