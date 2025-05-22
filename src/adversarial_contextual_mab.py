import numpy as np

from agents.agent import Agent, Exp3
from environments.adversarial_contextual_env import AdversarialContextualEnv
from src.adversaries.deceptive_adversary import DeceptiveAdversary
from src.agents.old_agents import LinEpsilonGreedy
from src.contexts.gaussian_context import GaussianContext
from src.display import display


class AdversarialContextualMAB:

    def __init__(self):
        pass 

    def play(self, agent: Agent, environment: AdversarialContextualEnv, num_sim: int, horizon: int):
        
        # init results
        rewards = np.zeros((num_sim, horizon))
        regrets = np.zeros((num_sim, horizon))
        pseudo_regrets = np.zeros((num_sim, horizon))
        avg_rewards = np.zeros((num_sim, horizon))

        # Iterate over simulations
        for n in range(num_sim):
            agent.reset()

            # Iterate over time steps
            for t in range(horizon):

                # Interact with the environment
                context = environment.get_context()
                # action_set = environment.get_action_set()
                action = agent.get_action()
                reward = environment.get_reward(action, context)
                agent.receive_reward(action,reward)

                # compute instantaneous reward  and (pseudo) regret
                rewards[n,t] = reward
                means = environment.get_mean_rewards()
                best_reward = np.max(means)
                regrets[n,t]= best_reward - reward # this can be negative due to the noise, but on average it's positive
                avg_rewards[n,t] = means[action]
                pseudo_regrets[n,t] = best_reward - means[action]

        return agent.name, rewards, regrets, avg_rewards, pseudo_regrets

#     Runnable for contextual bandit environments
if __name__ == "__main__":
    d = 7  # Feature dimension
    K = 2  # Number of arms per timestep (number of website versions)
    n_contexts = 2  # Number of contexts (types of users)
    theta = np.random.normal(0., 1., size=d)
    theta = theta / np.linalg.norm(theta)

    T = 1000  # Finite Horizon
    N = 50  # Monte Carlo simulations

    # The parameter theta is unknown, but we know it's been normalized (the l2 norm of theta is 1)
    # Feature vectors are also normalized
    # lin_env = LinearBandit(theta, K, n_contexts)

    # Save subsampled points for Figures
    Nsub = 100
    tsav = range(2, T, Nsub)
    # Choice of percentile display
    q = 10

    exp3 = Exp3(K)

    acmab = AdversarialContextualMAB()
    adversary = DeceptiveAdversary()
    context = GaussianContext(n_contexts, K, d)
    # Example contextual algorithm
    adversarial_env = AdversarialContextualEnv(adversary, context)
    exp3_experiment = acmab.play(exp3, adversarial_env, num_sim=N, horizon=T)
    # lin_eps_greedy = LinEpsilonGreedy(d, 1.0, 0.1)
    # lin_eps_greedy_experiment = acmab.play(lin_eps_greedy, adversarial_env , num_sim=N, horizon=T)
    # # display.plot_result(lin_eps_greedy_experiment)
