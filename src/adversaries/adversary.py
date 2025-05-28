import numpy as np

class Adversary:
    def __init__(self):
        self.history = []

    def update_history(self, action, reward):
        self.history.append((action, reward))

    def get_reward(self, action: int, context: np.ndarray) -> float:
        pass

    def get_best_reward(self):
        pass

    def get_mean_rewards(self, context: np.ndarray) -> np.ndarray:
        pass
    
    def reset(self):
        self.history = []