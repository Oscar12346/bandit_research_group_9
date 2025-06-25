import numpy as np
from src.adversaries.adversary import Adversary

class EntropyBasedAdversary(Adversary):
    def __init__(self, k: int, n: int, sigma: float = 0.1, delta: float = 0.05):
        """
        k      = number of arms
        n      = total number of rounds
        sigma  = standard deviation of the base Gaussian
        delta  = target probability (→ δ in the theorem)
        """
        self.k      = k
        self.n      = n
        self.sigma  = sigma
        self.delta  = delta

        # compute Δ = σ * sqrt{ (k−1)/(2n) ⋅ log(1/(8δ)) }
        self.Delta = sigma * np.sqrt((k - 1) / (2 * n) * np.log(1 / (8 * delta)))

        # fix arm 0 as the “baseline” (shift +Δ), pick best_arm ∈ {1,…,k−1}
        self.baseline_arm = 0
        self.best_arm     = np.random.randint(1, k)

        self.history = []           # (action, reward) pairs
        self._last_rewards = None   # stores the full reward vector for the current round

    def reset(self):
        """Start a fresh sequence."""
        self.history = []
        self._last_rewards = None

    def update_history(self, action: int, reward: float):
        self.history.append((action, reward))

    def get_reward(self, action: int, context=None) -> float:
        """
        Draws a new η_t drawn from N(0.5, alpha^2), clips it to [0,1], then
        applies the +Δ shifts:
          - arm 0         : η_t + Δ
          - self.best_arm : η_t + 2Δ
          - all others    : η_t
        Returns only the reward for `action` and stores the full vector.
        """
        eta = np.random.normal(loc=0.5, scale=self.sigma)

        # build all k rewards
        r = np.full(self.k, eta)
        r[self.baseline_arm]        = eta + self.Delta
        r[self.best_arm]            = eta + 2 * self.Delta
        r = np.minimum(1.0, np.maximum(0.0, r))

        self._last_rewards = r
        return float(r[action])

    def get_best_reward(self) -> float:
        """The reward you *could* have gotten this round (i.e. arm = self.best_arm)."""
        if self._last_rewards is None:
            raise RuntimeError("No round has been played yet.")
        return float(self._last_rewards[self.best_arm])

    def get_mean_rewards(self) -> np.ndarray:
        """
        Returns the entire reward vector for the last call to get_reward().
        (i.e. the “mean” reward of each arm under this adversary’s draw).
        """
        if self._last_rewards is None:
            raise RuntimeError("No round has been played yet.")
        return self._last_rewards.copy()
