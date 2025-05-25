import numpy as np


class Environment:

    def __init__(self):
        pass

    def get_reward(self, action: int) -> float:
        pass

    def get_mean_rewards(self, context: np.ndarray, action: int) -> np.ndarray:
        pass

    def get_adversary(self):
        pass