import numpy as np

from src.agents.agent import Agent

class LinExp3Agent(Agent):

    def __init__(self, name: str, K: int, d: int, sigma: np.ndarray, eta: float = 0.1, gamma: float = 0.1, epsilon: float = 0.0):
        self.name = name
        self.K = K
        self.d = d
        self.sigma = sigma
        self.sigma_inv = np.linalg.inv(sigma)
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Cumulative estimators Theta for each arm (K x d)
        self.theta = np.zeros((self.K, self.d))
        self.current_context = None
        self.last_policy = None
        self.last_action = None
        
    def reset(self):
        self.theta = np.zeros((self.K, self.d))
        self.current_context = None
        self.last_policy = None
        self.last_action = None

    def set_context(self, context: np.ndarray):
        self.current_context = context

    def get_policy(self, ctx_t: np.ndarray) -> np.ndarray:
        scores = -self.eta * np.dot(self.theta, ctx_t)  

        # Stabilize: subtract max for numerical stability
        max_score = np.max(scores)
        stable_scores = scores - max_score
        exp_scores = np.exp(stable_scores)

        # fallback: uniform policy
        weights_sum = np.sum(exp_scores)
        if weights_sum <= 0 or np.isnan(weights_sum):
            norm_weights = np.ones(self.K) / self.K
        else:
            norm_weights = exp_scores / weights_sum

        # Policy calculation
        policy = (1 - self.gamma) * norm_weights + self.gamma / self.K
        return policy

    def get_action(self, context: np.ndarray) -> int:
        self.set_context(context)
        self.last_policy = self.get_policy(self.current_context)
        self.last_action = np.random.choice(self.K, p=self.last_policy)
        return self.last_action

    def receive_reward(self, action: int, reward: float):
        pass
