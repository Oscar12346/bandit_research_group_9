import numpy as np

class Agent:

    def __init__(self, name: str):
        self.name = name

    def reset(self):
        pass

    def get_action(self, context: np.ndarray) -> int:
        pass

    def receive_reward(self, action: int, reward: float):
        pass