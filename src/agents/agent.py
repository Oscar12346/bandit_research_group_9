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
    def __init__(self, K, horizon):
        lr = np.sqrt(np.log(K) / (horizon * K))
        self.lr = lr  # learning rate
        self.K = K  # arms
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



# TODO cleaning up + testing (rn still bit of a mess)
class RobustLinExp3(Agent):
    def __init__(self, d, K, horizon, lr=0.1, gamma=0.1, covariance=None):
        # TODO initialize lr and gamma as in paper
        self.gamma = gamma
        self.lr = lr
        self.d = d
        self.K = K
        if covariance is None:
            self.covariance = np.identity(self.d)
            self.invcov = np.identity(self.d)
        else:
            self.covariance = covariance
            self.invcov = np.linalg.inv(covariance)

        self.reset()

    def reset(self):
        # Copied this from LinGreedy
        self.t = 0
        self.theta_hat = np.zeros(self.d)

        # The covariance matrix is initialized here
        self.covariance = np.identity(self.d)

        # The inverse of the covariance matrix is initialized here
        self.invcov = np.identity(self.d)

        self.last_action_index = None
        self.last_probs = None
        self.last_context = None

    def get_action(self, arms, context):
        K, _ = arms.shape
        self.last_context = context
        weights = np.zeros(K, dtype=float)
        # summation in weighting is done by updating theta_hat with sum of old value + new
        for a in weights:
            weights[a] = np.exp(-self.lr * np.dot(context, self.theta_hat[a]))

        weights /= np.sum(weights)

        probs = (1 - self.gamma) * weights + self.gamma/K
        self.last_probs = probs
        chosen_index = np.random.choice(K, p=probs)
        self.last_action_index = chosen_index

        return arms[chosen_index]


    def receive_reward(self, chosen_arm, reward):
        loss = 1 - reward #TODO ?couldn't really find how paper defines loss without true params
        x = self.last_context
        p = self.last_probs[chosen_arm]
        self.theta_hat += (1 / p) * np.dot(self.invcov, x) * loss # update the least square estimate
        self.t += 1

# TODO
class RealLinExp3(RobustLinExp3):
    def __init__(self, d, K, horizon, lr=0.1, gamma=0.1, covariance=None):
        super().__init__(d, K, horizon, lr, gamma, covariance)
    def get_action(self, arms, context):
        ...
    def receive_reward(self, chosen_arm, reward):
        ...


