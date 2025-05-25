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