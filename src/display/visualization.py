import numpy as np
import matplotlib.pyplot as plt

class Visualization:

    @staticmethod
    def plot_mean_std(
        data: np.ndarray, 
        title: str = "", 
        xlabel: str = "",
        ylabel: str = "",
        baseline: np.ndarray = None,
        baseline_title: str = "Baseline"
    ):
        mean_values = np.mean(data, axis=0)
        std_values = np.std(data, axis=0)
        timesteps = np.arange(data.shape[1])

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, mean_values, label='Mean')
        plt.fill_between(
            timesteps, 
            mean_values - std_values, 
            mean_values + std_values,
            color='b', 
            alpha=0.2, 
            label='±1 Std. Dev.'
        )

        # Plot baseline if provided
        if baseline is not None:
            baseline = baseline.flatten()  # ensure 1D
            if baseline.shape[0] != data.shape[1]:
                raise ValueError(f"Baseline length {baseline.shape[0]} does not match data horizon {data.shape[1]}")
            plt.plot(timesteps, baseline, label=baseline_title, color='r', linestyle='--')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
