import numpy as np

from src.adversaries.adversary import Adversary
from src.environments.environment import Environment

class Adversarial_MAB_env(Environment):

    def __init__(self, K: int, adversary: Adversary):
        self.K = K
        self.adversary = adversary

    def get_reward(self, action):
        return self.adversary.get_reward(action, None)
    
    def get_mean_rewards(self):
        return self.adversary.get_mean_rewards()

    def get_adversary(self):
        return self.adversary

class Random_Adversarial_MAB_env(Environment):

    def __init__(self, arms: int, r_range=None):
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