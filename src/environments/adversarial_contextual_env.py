import numpy as np

from src.adversaries.adversary import Adversary
from src.environments.environment import Environment
from src.contexts.context import Context

class AdversarialContextualEnv(Environment):

    def __init__(self, adversary: Adversary, context: Context):

        self.adversary = adversary
        self.context = context
    
    def get_reward(self, action: int, context: np.ndarray) -> float:
        return self.adversary.get_reward(action, context)
    
    def get_context(self) -> np.ndarray:
        return self.context.get_context()
    
    def get_mean_rewards(self, context: np.ndarray, action: int) -> np.ndarray:
        return self.adversary.get_mean_rewards(context, action)