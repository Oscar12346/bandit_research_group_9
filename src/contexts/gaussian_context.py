import numpy as np

from src.contexts.context import Context

class GaussianContext(Context):

    def __init__(self, n_contexts: int, n_arms: int, n_features: int):
        
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.n_features = n_features
        
        # Generate random features for each context
        self.features = np.random.multivariate_normal(
            np.zeros(self.n_features), 
            np.eye(self.n_features), 
            size=(self.n_contexts, self.n_arms)
        )

        # Normalize the features
        norms = np.linalg.norm(self.features, axis=2)
        self.features = self.features / norms[:, :, np.newaxis]

        # Return action set
        current_context_id = np.random.randint(self.n_contexts)
        self.current_action_set = self.features[current_context_id, :, :]

    def get_context(self) -> np.ndarray:
        
        current_context_id = np.random.randint(self.n_contexts)
        return self.features[current_context_id]

    def get_action_set(self):
        """Returns the set of available arms (feature vectors) for the current round"""
        current_context_id = np.random.randint(self.n_contexts)
        self.current_action_set = self.features[current_context_id, :, :]
        return self.current_action_set
