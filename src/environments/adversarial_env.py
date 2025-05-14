
from adversaries.adversary import Adversary
from environments.environment import Environment

class AdversarialEnv(Environment):

    def __init__(self, adversary: Adversary):

        self.adversary = adversary
    
    def get_reward(self, action: int) -> float:
        return self.adversary.get_reward(action)


