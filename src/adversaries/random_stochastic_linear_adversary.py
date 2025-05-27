import numpy as np

class RandomStochasticLinearAdversary:
    def __init__(self, num_actions: int, context_dim: int, horizon: int, noise_mean: float = 0.0, noise_std: float = 0.0):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.t = 0  # current timestep

        # Generate a different theta_t,a for each timestep and action
        self.theta_sequence = np.random.randn(horizon, num_actions, context_dim)

    def get_reward(self, action: int, context: np.ndarray) -> float:
        """Returns reward for a given action and context at time t."""
        theta = self.theta_sequence[self.t, action]
        reward = np.dot(context, theta)
        reward += np.random.normal(self.noise_mean, self.noise_std)

        self.step()
        return reward

    def get_best_reward(self, context: np.ndarray) -> float:
        """Returns the best possible reward for current context."""
        thetas = self.theta_sequence[self.t]
        rewards = thetas @ context
        return np.max(rewards)

    def get_mean_rewards(self, context: np.ndarray) -> np.ndarray:
        """Returns mean reward for each action for a given context."""
        thetas = self.theta_sequence[self.t]
        return thetas @ context

    def reset(self):
        """Resets the adversary for a new episode."""
        self.t = 0

    def step(self):
        """Advance time (e.g., after each round)."""
        self.t = min(self.t + 1, self.horizon - 1)
