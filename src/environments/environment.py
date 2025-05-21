import numpy as np


class Environment:

    def __init__(self):
        pass

    def get_reward(self, action: int) -> float:
        pass

    def get_mean_rewards(self) -> list:
        pass


class MAB_env(Environment):

    def __init__(self, means, noise=1.):
        """
        theta: d-dimensional vector (bounded) representing the hidden parameter
        K: number of actions per round (random action vectors generated each time)
        """
        self.theta = means
        self.noise = noise
        self.K = np.size(means)

    def get_reward(self, action):
        """ sample reward given action
        """
        return np.random.normal(self.theta[action], self.noise)

    def get_means(self):
        return self.theta

class Random_Adversarial_MAB_env(Environment):

    def __init__(self, arms, r_range=None):
        if r_range is None:
            r_range = [0.0, 1.0]
        self.r_range = r_range
        self.K = arms
        self.theta = np.zeros(arms)

    def get_reward(self, action):
        self.theta = np.random.uniform(self.r_range[0], self.r_range[1], self.K)
        return self.theta[action]

    def get_means(self):
        return self.theta