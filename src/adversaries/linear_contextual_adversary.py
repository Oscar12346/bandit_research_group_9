import numpy as np


class ContextualLinearAdversary:
    def __init__(
            self,
            num_actions: int,
            context_dim: int,
            horizon,
    ):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.t = 0

        self.horizon = horizon

        theta0 = np.random.randn(num_actions, context_dim)
        norms0 = np.linalg.norm(theta0, axis=1, keepdims=True)
        normalized_theta = theta0 / np.maximum(norms0, 1e-8)  # avoid divide by zero

        scaling_factors = np.linspace(norms0, 1.0, horizon).reshape(horizon, num_actions, 1)

        self.theta = normalized_theta[np.newaxis, :, :] * scaling_factors

        print(self.theta)


    def reset(self):
        """Resets the adversary for a new episode."""
        theta0 = np.random.randn(self.num_actions, self.context_dim)
        norms0 = np.linalg.norm(theta0, axis=1, keepdims=True)
        normalized_theta = theta0 / np.maximum(norms0, 1e-8)  # avoid divide by zero

        scaling_factors = np.linspace(norms0, 1.0, self.horizon).reshape(self.horizon, self.num_actions, 1)

        self.theta = normalized_theta[np.newaxis, :, :] * scaling_factors
        self.t = 0

    def get_reward(self, action: int, context: np.ndarray) -> float:
        """Returns reward for a given action and context at time t."""
        theta = self.theta[self.t,action]
        reward = theta @ context
        # Clip the reward to be in the range [-1, 1]
        reward = max(min(reward, 1), -1)

        self.t += 1

        return reward

    def get_best_reward(self, context: np.ndarray) -> float:
        """Returns the best possible reward for current context."""
        rewards = self.theta[self.t-1] @ context
        return np.max(rewards)

    def get_mean_rewards(self, context: np.ndarray) -> np.ndarray:
        """Returns mean reward for each action for a given context."""
        return self.theta[self.t - 1] @ context

