import numpy as np

from adversaries.adversary import Adversary
from environments.environment import Environment
from contexts.context import Context

class AdversarialContextualEnv(Environment):

    def __init__(self, adversary: Adversary, context: Context):

        self.adversary = adversary
        self.context = context
    
    def get_reward(self, action: int, context: np.ndarray) -> float:
        return self.adversary.get_reward(action, context)
    
    def get_context(self) -> np.ndarray:
        return self.context.get_context()