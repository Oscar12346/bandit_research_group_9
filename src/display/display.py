import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
plt.rcParams["animation.html"] = "jshtml"

import seaborn as sns
colors = sns.color_palette('colorblind')

def plot_result(regrets, logscale=False, lb=None, q=10, mode='regret', cumulative=True, bounds_per_timestep = None):
    """
    regrets must be a dict {'agent_id':regret_table}
    """

    reg_plot = plt.figure()
    # compute useful stats
    #     regret_stats = {}
    for i, agent_id in enumerate(regrets.keys()):
        data = regrets[agent_id]
        N, T = data.shape
        if cumulative:
            cumdata = np.cumsum(data, axis=1)  # cumulative regret
        else:
            cumdata = data

        mean_reg = np.mean(cumdata, axis=0)
        q_reg = np.percentile(cumdata, q, axis=0)
        Q_reg = np.percentile(cumdata, 100 - q, axis=0)

        #         regret_stats[agent_id] = np.array(mean_reg, q_reg, Q_reg)

        plt.plot(np.arange(T), mean_reg, color=colors[i], label=agent_id)
        plt.fill_between(np.arange(T), q_reg, Q_reg, color=colors[i], alpha=0.2)

    if logscale:
        plt.xscale('log')
        plt.xlim(left=100)

    if lb is not None:
        plt.plot(np.arange(T), lb, color='black', marker='*', markevery=int(T / 10))

    if bounds_per_timestep is not None:
        plt.plot(np.arange(T), bounds_per_timestep, color='gray', linestyle='--', label='Theoretical Bound')

    plt.xlabel('time steps')
    header = "Cumulative " if cumulative else ""
    plt.ylabel(header + mode.capitalize())
    plt.legend()
    reg_plot.show()
    return reg_plot  # notebook version