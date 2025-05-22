from numpy import ndarray

from src.adversaries.adversary import Adversary


class DeceptiveAdversary(Adversary):
    def __init__(self):
        super().__init__()

        self.history = []

    def update_history(self, action, reward):
        self.history.append((action, reward))

    def get_reward(self, action: int, context: ndarray) -> float:
        t = len(self.history)

        # Implementation now expects that there are 2 actions,
        # when we extend this Adversary to have knowledge about the amount of actions we can extend
        if t < 500:
            # Make it seem that action 0 is really good and the rest worse in the first 500 timesteps
            reward = 1.0 if action == 0 else 0.25
            self.update_history(action, reward)
            return reward
        else:
            # Now we expose which action is actually the best, which is action 1
            # All other actions receive no reward anymore
            reward = 1.0 if action == 1 else 0.0
            self.update_history(action, reward)
            return reward


    def get_best_reward(self):
        return 1.0