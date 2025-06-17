import numpy as np
from src.adversaries.adversary import Adversary

class ContextualProbabilisticAdversary(Adversary):
    """Contextual Probabilistic Adversary, that assumes it may know the context before it chooses the lost vectors"""
    def __init__(self, num_actions: int, context_dim: int, horizon: int, eta=0.1, gamma=0.1, sigma=None, ub=0.3):
        super().__init__()
        self.K = num_actions
        self.T = horizon
        self.d = context_dim
        self.eta = eta
        self.gamma = gamma

        self.theta_sequence = np.random.randn(self.T, self.K, self.d)
        norms = np.linalg.norm(self.theta_sequence, axis=2, keepdims=True)
        self.theta_sequence = self.theta_sequence / np.maximum(norms, 1.0)

        self.sigma = sigma if sigma is not None else np.identity(self.d)
        self.sigma_inv = np.linalg.inv(self.sigma)

        self.rewards = np.zeros((self.K, self.T)) - 1
        self.current_context = None
        self.best_arm = 0
        self.mean_rewards = np.zeros(self.K)
        self.theta_hat = np.zeros((self.K, self.d))
        self.ub = ub

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

        self.current_context = context

        if t < self.T - 1:
            policy = self.estimate_policy(context, t)

            sorted_indices = np.argsort(-policy)
            cumulative_prob = 0.0
            high_loss_actions = []
            low_loss_actions = []

            for i in sorted_indices:
                if cumulative_prob < self.ub:
                    high_loss_actions.append(i)
                    cumulative_prob += policy[i]
                else:
                    low_loss_actions.append(i)

            # Assign rewards: -1 for high-loss (likely) actions, +1 for low-loss
            desired_rewards = np.zeros(self.K)
            desired_rewards[high_loss_actions] = -1.0
            desired_rewards[low_loss_actions] = 1.0

            # Construct theta so that -theta @ context ≈ desired_reward
            context_unit = context / (np.linalg.norm(context) + 1e-8)
            for a in range(self.K):
                magnitude = -desired_rewards[a] / (np.linalg.norm(context) + 1e-8)
                self.theta_sequence[t, a] = magnitude * context_unit

            self.observe_reward(action, policy)

        reward = -self.theta_sequence[t, action] @ context
        reward = np.clip(reward, -1, 1)


        return reward


    def observe_reward(self, a: int, policy: np.ndarray):
        """
        Called after agent selects an action, with its policy distribution.
        This function updates theta_sequence[t] and cumulative theta_hat.
        """
        t = len(self.history) - 1  # Just finished round t
        context = self.current_context
        if context is None:
            raise ValueError("Context not set before calling observe_reward")

        # For the selected action a_t, construct estimator update

        pi_a = policy[a]
        eps = 1e-5
        # Loss estimator (importance-weighted)
        loss_estimate = - (self.theta_sequence[t, a] @ context)
        estimator = (1 / (pi_a + eps)) * loss_estimate * (self.sigma_inv @ context)

        # Update theta_hat (for tracking)
        self.theta_hat[a] += estimator


    def get_best_reward(self):
        t = len(self.history) -1 # history already updated when this is called
        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
        return self.rewards[self.best_arm, t]

    def get_mean_rewards(self, context: np.ndarray) -> np.ndarray:
        t = min(len(self.history), self.T - 1)
        theta_t = self.theta_sequence[t]
        return theta_t @ context  # Shape: (K,)

    def get_loss_vectors(self):
        return self.theta_sequence

    def reset(self):
        super().reset()
        self.theta_sequence = np.random.randn(self.T, self.K, self.d)
        norms = np.linalg.norm(self.theta_sequence, axis=2, keepdims=True)
        self.theta_sequence = self.theta_sequence / np.maximum(norms, 1.0)

        self.rewards = np.zeros((self.K, self.T))
        self.current_context = None
        self.best_arm = 0

class ContextualTargetedProbabilisticAdversary(Adversary):
    """Uses previous context to estimate policy for current context and updates loss vectors based on those"""
    def __init__(self, num_actions: int, context_dim: int, horizon: int, eta=0.1, gamma=0.1, sigma=None, ub=0.3,
                 var=0.3):
        super().__init__()
        self.ub = ub
        self.var = var
        self.K = num_actions
        self.T = horizon
        self.d = context_dim
        self.eta = eta
        self.gamma = gamma

        self.theta_sequence = np.random.randn(self.T, self.K, self.d)
        norms = np.linalg.norm(self.theta_sequence, axis=2, keepdims=True)
        self.theta_sequence = self.theta_sequence / np.maximum(norms, 1.0)

        self.sigma = sigma if sigma is not None else np.identity(self.d)
        self.sigma_inv = np.linalg.inv(self.sigma)

        self.rewards = np.zeros((self.K, self.T)) - 1
        self.last_context = None
        self.best_arm = 0
        self.mean_rewards = np.zeros(self.K)
        self.used_loss_vectors = []
        self.theta_hat = np.zeros((self.K, self.d))

    def estimate_policy(self, t: int) -> np.ndarray:
        theta_t = self.theta_hat
        scores = -self.eta * (theta_t @ self.last_context)
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
        t = len(self.history)

        if t >= self.T:
            raise ValueError("We shouldnt get here")

        if t == 0:
            reward = - np.clip(self.theta_sequence[t, action] @ context, -1, 1)
            self.update_history(action, reward)
            self.last_context = context
            return reward

        policy = self.estimate_policy(t)

        theta_t = self.theta_sequence[t]

        # for a in range(self.K):
        #     prob = policy[a]
        #     direction = 1.0 if prob >= self.ub else -1.0  # more loss if likely
        #     delta = direction * self.eta * self.last_context  # eta controls magnitude
        #     theta_t[a] += delta
        #
        # # Compute base rewards from the linear model
        #
        #
        # # print(f'theta_t: {self.theta_sequence[t]}')
        # norms = np.linalg.norm(theta_t, axis=1, keepdims=True)
        # self.theta_sequence[t] = theta_t / np.maximum(norms, 1.0)

        sorted_indices = np.argsort(-policy)
        cumulative_prob = 0.0
        high_loss_actions = []
        low_loss_actions = []

        for i in sorted_indices:
            if cumulative_prob < self.ub:
                high_loss_actions.append(i)
                cumulative_prob += policy[i]
            else:
                low_loss_actions.append(i)

        desired_rewards = np.zeros(self.K)
        desired_rewards[high_loss_actions] = 1.0
        desired_rewards[low_loss_actions] = -1.0

        context_unit = self.last_context / (np.linalg.norm(self.last_context) + 1e-8)
        for a in range(self.K):
            magnitude = -desired_rewards[a] / (np.linalg.norm(self.last_context) + 1e-8)
            self.theta_sequence[t, a] = magnitude * context_unit


        reward = - np.clip(self.theta_sequence[t, action] @ context, -1, 1)
        # print(f'action: {action}, reward: {reward}')
        self.last_context = context
        self.observe_reward(action, policy)
        self.update_history(action, reward)
        return reward

    def observe_reward(self, a: int, policy: np.ndarray):
        """
        Called after agent selects an action, with its policy distribution.
        This function updates theta_sequence[t] and cumulative theta_hat.
        """
        t = len(self.history) - 1  # Just finished round t
        context = self.last_context
        if context is None:
            raise ValueError("Context not set before calling observe_reward")

        # For the selected action a_t, construct estimator update

        pi_a = policy[a]
        eps = 1e-5
        # Loss estimator (importance-weighted)
        loss_estimate = - (self.theta_sequence[t, a] @ context)
        estimator = (1 / (pi_a + eps)) * loss_estimate * (self.sigma_inv @ context)

        # Update theta_hat (for tracking)
        self.theta_hat[a] += estimator


    def get_best_reward(self):
        t = len(self.history) -1 # history already updated when this is called
        self.best_arm = np.argmax(np.sum(self.rewards, axis=1))
        return self.rewards[self.best_arm, t]

    def get_mean_rewards(self, context: np.ndarray) -> np.ndarray:
        t = min(len(self.history), self.T - 1)
        theta_t = self.theta_sequence[t]
        return theta_t @ context  # Shape: (K,)

    def get_loss_vectors(self):
        t = len(self.history)
        return self.theta_sequence

    def reset(self):
        super().reset()
        self.theta_sequence = np.random.randn(self.T, self.K, self.d)
        norms = np.linalg.norm(self.theta_sequence, axis=2, keepdims=True)
        self.theta_sequence = self.theta_sequence / np.maximum(norms, 1.0)

        self.rewards = np.zeros((self.K, self.T))
        self.last_context = None
        self.best_arm = 0
        self.theta_hat = np.zeros((self.K, self.d))