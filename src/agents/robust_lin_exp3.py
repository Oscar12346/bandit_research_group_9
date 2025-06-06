import numpy as np
from src.agents.lin_exp3 import LinExp3Agent

class RobustLinExp3Agent(LinExp3Agent):

    def receive_reward(self, action, reward):

        loss = -reward
        pi_a = self.last_policy[action]
        x = self.current_context

        # Calculate estimator theta_hat for chosen action only
        theta_hat = (1 / pi_a) * (self.sigma_inv @ x) * loss

        # Update cumulative estimator for chosen action
        self.theta[action] += theta_hat

    def compute_regret_bound(self, T: int) -> float:
        K = self.K
        d = self.d
        eta = T**(-2/3)*(K*d)**(-1/3)*np.log(K)**(2/3) if T > 0 else 1.0
        gamma = self.gamma
        epsilon = self.epsilon

        term1 = 2 * np.sqrt(d) * epsilon * T
        term2 = 2 * gamma * T
        term3 = (2 * eta * K * d * T) / gamma
        term4 = (np.log(K)) / eta

        regret_bound = term1 + term2 + term3 + term4
        return regret_bound