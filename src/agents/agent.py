import numpy as np


class Agent:

    def __init__(self, name: str):
        self.name = name

    def reset(self):
        pass

    def get_action(self):
        pass

    def receive_reward(self, action: int, reward: float):
        pass

class Exp3(Agent):
  def __init__(self, K, lr=0.1):
      self.lr = lr # learning rate
      self.K = K # arms
      self.estimated_rewards = np.zeros(K)  # S_hat from paper
      # Initialize Probability distribution (P_t)
      self.prop_dist = np.ones(K, dtype=float) / self.K
      self.t = 0
      self.last_action = None
      self.reset()

  def reset(self):
      self.estimated_rewards = np.zeros(self.K)
      self.t = 0

  def get_action(self):
      # Update probability distribution
      weights = np.exp(self.estimated_rewards * self.lr)
      self.probs = weights / np.sum(weights)
      # Sample arm from the probability distribution
      action = np.random.choice(self.K, p=self.probs)
      self.last_action = action  # store for update
      return action

  def receive_reward(self, chosen_arm, reward):
      P_ti = self.prop_dist[chosen_arm]
      # print(reward)
      estimated_reward = (1 - reward) / P_ti
      # Update S_hat as per: Ŝ_ti = Ŝ_{t−1,i} + 1 - I{At=i} * (Xt / Pt[i])
      self.estimated_rewards[chosen_arm] += 1 - estimated_reward
      self.t += 1

  def name(self):
      return f'Exp3(lr={self.lr})'