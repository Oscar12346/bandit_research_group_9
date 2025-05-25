import numpy as np

class Adversary:
    def __init__(self):
        self.history = []

    def update_history(self, action, reward):
        self.history.append((action, reward))

    def get_reward(self, action: int, context: np.ndarray) -> float:
        pass

    def get_best_reward(self, t: int):
        pass

    def reset(self):
        self.history = []