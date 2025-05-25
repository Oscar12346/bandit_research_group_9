import numpy as np

from src.adversaries.adversary import Adversary

class StochasticAdversary(Adversary):
    def __init__(self, K, T):
        super().__init__()

        rng = np.random.default_rng(seed=42)

        self.rewards = rng.uniform(low=0.0, high=1.0, size=(K, T))
        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))

    def get_reward(self, action: int, context: np.ndarray) -> float:
        t = len(self.history)

        reward = self.rewards[action, t]
        self.update_history(action, reward)
        return reward
    
    def get_best_reward(self, t: int):
        return self.rewards[self.best_arm, t]
    