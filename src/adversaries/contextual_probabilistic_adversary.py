import numpy as np
from src.adversaries.adversary import Adversary

class ContextualProbabilisticAdversary(Adversary):
    def __init__(self, num_actions: int, context_dim: int, horizon: int, eta=0.1, gamma=0.1, ub= 0.3, sigma=None):
        super().__init__()
        self.K = num_actions
        self.T = horizon
        self.d = context_dim
        self.eta = eta
        self.gamma = gamma
        self.ub = ub

        self.theta_sequence = np.random.randn(self.T, self.K, self.d)
        norms = np.linalg.norm(self.theta_sequence, axis=2, keepdims=True)
        self.theta_sequence = self.theta_sequence / np.maximum(norms, 1.0)

        self.sigma = sigma if sigma is not None else np.identity(self.d)
        self.sigma_inv = np.linalg.inv(self.sigma)

        self.rewards = np.zeros((self.K, self.T))
        self.current_context = None
        self.best_arm = 0

    def estimate_policy(self, context: np.ndarray, t: int) -> np.ndarray:
        theta_t = self.theta_sequence[t]
        scores = -self.eta * (theta_t @ context)
        stable_scores = scores - np.max(scores)
        exp_scores = np.exp(stable_scores)
        weights_sum = np.sum(exp_scores)

        if weights_sum <= 0 or np.isnan(weights_sum):
            norm_weights = np.ones(self.K) / self.K
        else:
            norm_weights = exp_scores / weights_sum

        policy = (1 - self.gamma) * norm_weights + self.gamma / self.K
        return policy

    def get_reward(self, action: int, context: np.ndarray) -> float:
        """Returns reward for a given action and context at time t."""
        t = len(self.history)
        theta = self.theta_sequence[t, action]
        reward = np.dot(context, theta)
        self.current_context = context

        # Clip the reward to be in the range [-1, 1]
        reward = max(min(reward, 1), -1)
        self.update_history(action, reward)

        if t < self.T - 1:
            policy = self.estimate_policy(context, t)
            self._update_future_rewards(t + 1, policy)

        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
        return reward

    # def _update_future_rewards(self, t_next: int, policy: np.ndarray):
    #     sorted_indices = np.argsort(policy)
    #     cumulative_prob = 0.0
    #     reward_indices = []
    #
    #     for i in sorted_indices:
    #         p = policy[i]
    #         if cumulative_prob + p > self.ub:
    #             if len(reward_indices) == 0:
    #                 reward_indices.append(i)
    #             break
    #         if cumulative_prob + p <= self.ub:
    #             reward_indices.append(i)
    #             cumulative_prob += p
    #         else:
    #             break
    #
    #     self.rewards[:, t_next] = -1
    #     self.rewards[reward_indices, t_next] = 1

    def _update_future_rewards(self, t_next: int, policy: np.ndarray):
        theta_t = self.theta_sequence[t_next]  # shape: (K, d)
        context = self.current_context

        if context is None:
            raise ValueError("Context is None in _update_future_rewards")

        context = np.asarray(context).reshape(-1)  # shape: (d,)
        if context.shape[0] != self.d:
            raise ValueError(f"Context shape mismatch: expected {self.d}, got {context.shape}")

        # Compute base rewards from the linear model
        base_rewards = theta_t @ context  # shape: (K,)

        # Adversarial twist: reward unlikely actions more
        rewards = -policy * base_rewards

        # Normalize to [-1, 1]
        min_r, max_r = rewards.min(), rewards.max()
        if max_r > min_r:
            normalized_rewards = 2 * (rewards - min_r) / (max_r - min_r) - 1
        else:
            normalized_rewards = np.zeros_like(rewards)

        self.rewards[:, t_next] = normalized_rewards


    def observe_reward(self, action: int, reward: float):
        # Estimate the reward using a model or heuristics
        t = len(self.history)
        pi_a = self.estimate_policy(self.current_context, t)[action]

        theta_t = self.theta_sequence[t]
        estimated_reward = theta_t[action] @ self.current_context

        estimated_reward = np.clip(estimated_reward, -1, 1)

        loss = -estimated_reward
        theta_hat = (1 / pi_a) * (self.sigma_inv @ self.current_context) * loss
        self.theta_sequence[t, action] += theta_hat


    def get_best_reward(self):
        t = len(self.history) -1 # history already updated when this is called
        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
        return self.rewards[self.best_arm, t]

    def get_mean_rewards(self, context: np.ndarray) -> np.ndarray:
        t = min(len(self.history), self.T - 1)
        theta_t = self.theta_sequence[t]
        return theta_t @ context  # Shape: (K,)

    def reset(self):
        super().reset()
        self.theta_sequence = np.random.randn(self.T, self.K, self.d)
        norms = np.linalg.norm(self.theta_sequence, axis=2, keepdims=True)
        self.theta_sequence = self.theta_sequence / np.maximum(norms, 1.0)

        self.rewards = np.zeros((self.K, self.T))
        self.current_context = None
        self.best_arm = 0
