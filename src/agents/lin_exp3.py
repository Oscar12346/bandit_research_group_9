import numpy as np

from src.agents.agent import Agent

class LinExp3Agent(Agent):

    def __init__(self, name: str, K: int, d: int, sigma: np.ndarray, eta: float = 0.1, gamma: float = 0.1):
        self.name = name
        self.K = K
        self.d = d
        self.sigma = sigma
        self.sigma_inv = np.linalg.inv(sigma)
        self.eta = eta
        self.gamma = gamma
        
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

    def compute_weight(self, ctx_t: np.ndarray, action: int) -> float:
        dot_product = np.dot(ctx_t, self.theta[action])
        return np.exp(-self.eta * dot_product)
    
    def get_policy(self, ctx_t: np.ndarray) -> np.ndarray:
        weights = np.array([self.compute_weight(ctx_t, a) for a in range(self.K)])
        weights_sum = weights.sum()
        norm_weights = weights / weights_sum
        policy = (1 - self.gamma) * norm_weights + self.gamma / self.K
        return policy

    def get_action(self, context: np.ndarray) -> int:
        self.set_context(context)
        self.last_policy = self.get_policy(self.current_context)
        self.last_action = np.random.choice(self.K, p=self.last_policy)
        return self.last_action

    def receive_reward(self, action: int, reward: float):
        pass
