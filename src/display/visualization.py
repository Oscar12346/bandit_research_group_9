import numpy as np
import matplotlib.pyplot as plt

class Visualization:

    @staticmethod
    def plot_mean_std(
        data: list[np.ndarray],
        data_labels: list[str] = None,
        title: str = "", 
        xlabel: str = "",
        ylabel: str = "",
        baseline: np.ndarray = None,
        baseline_title: str = "Baseline"
    ):
        
        if len(data) != len(data_labels):
            raise ValueError("data and data_labels must have the same length")
        
        # Get mean and std for each dataset
        mean_values = []
        std_values = []
        for i in range(len(data)):
            mean_values.append(np.mean(data[i], axis=0))
            std_values.append(np.std(data[i], axis=0))

        # Get the number of timesteps
        timesteps = np.arange(data[0].shape[1])

        # Plot each dataset with its mean and std
        plt.figure(figsize=(10, 6))
        for i in range(len(data)):
            plt.plot(timesteps, mean_values[i], label=data_labels[i])
            plt.fill_between(
                timesteps, 
                mean_values[i] - std_values[i], 
                mean_values[i] + std_values[i],
                alpha=0.2
            )

        # Plot baseline if provided
        if baseline is not None:
            baseline = baseline.flatten()  # ensure 1D
            if baseline.shape[0] != data[0].shape[1]:
                raise ValueError(f"Baseline length {baseline.shape[0]} does not match data horizon {data[0].shape[1]}")
            plt.plot(timesteps, baseline, label=baseline_title, color='r', linestyle='--')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
