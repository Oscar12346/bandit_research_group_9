import numpy as np

from src.adversaries.adversary import Adversary


class TitForTatAdversary(Adversary):
    def __init__(self, K):
        super().__init__()

        self.K = K
        self.take_arm = np.random.choice(K)
        self.split_arm = self.switch_arm = np.random.choice(np.setdiff1d(range(self.K), self.take_arm))

    def get_reward(self, action: int, context: np.ndarray) -> float:
        if len(self.history) > 0:
            previous_action = self.history[-1][0]
        else:
            previous_action = self.split_arm
        reward = self.get_arm_reward(action, previous_action)
        self.update_history(action, reward)
        return reward

    def get_arm_reward(self, action: int, prev_action: int) -> float:
        if prev_action == self.take_arm:
            return 0
        if prev_action == self.split_arm:
            return 0.6 if action == self.split_arm else 1

    def get_best_reward(self, t: int):
        if t != 0:
            previous_action = self.history[t-1][0]
            if previous_action == self.take_arm:
                return self.get_arm_reward(self.split_arm, previous_action)
            else:
                return self.get_arm_reward(self.take_arm, self.split_arm)
        return self.get_arm_reward(self.take_arm, self.split_arm)

    def get_mean_rewards(self, context: np.ndarray) -> np.ndarray:
        pass