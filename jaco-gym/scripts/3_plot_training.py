import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy


timesteps = 1e10
log_dir = "/home/rl/results/10/"

W = load_results(log_dir)
print(W)
# print("results: ", W)


# plot all training rewards
results_plotter.plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "")
plt.savefig(log_dir+"reward_vs_timesteps.png")
# plt.show()

results_plotter.plot_results([log_dir], timesteps, results_plotter.X_EPISODES, "")
plt.savefig(log_dir+"reward_vs_episodes.png")
# plt.show()

results_plotter.plot_results([log_dir], timesteps, results_plotter.X_WALLTIME, "")
plt.savefig(log_dir+"reward_vs_walltime.png")
# plt.show()


#### smoothed training rewards


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def moving_min(values, window):
    output=np.zeros(len(values)-window)
    for i in range(len(values)-window):
        output[i]=min(values[i:i+window])
    return output

def plot_results(log_folder, type_str):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param type: (str) either 'timesteps', 'episodes' or 'walltime_hrs'
    """

    x, y = ts2xy(load_results(log_folder), type_str)
    # print(x, y)

    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]
    # print(x, y)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(type_str)
    plt.ylabel('Rewards')

def plot_steps(log_folder):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param type: (str) either 'timesteps', 'episodes' or 'walltime_hrs'
    """

    x, y = ts2xy(load_results(log_folder), "timesteps")
    # print(x, y)

    y = moving_average(W["l"], window=50)
    # y = moving_min(W["l"], window=100)
    # Truncate x
    x = x[len(x) - len(y):]
    # print(x, y)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("timesteps")
    plt.ylabel('step')

def plot_steps_min(log_folder):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param type: (str) either 'timesteps', 'episodes' or 'walltime_hrs'
    """

    x, y = ts2xy(load_results(log_folder), "timesteps")
    # print(x, y)

    # y = moving_average(W["l"], window=100)
    y = moving_min(W["l"], window=50)
    # Truncate x
    x = x[len(x) - len(y):]
    # print(x, y)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("timesteps")
    plt.ylabel('step')


plot_results(log_dir, 'timesteps')
plt.savefig(log_dir+"reward_vs_timesteps_smoothed.png")
# plt.show()

plot_results(log_dir, 'episodes')
plt.savefig(log_dir+"reward_vs_episodes_smoothed.png")
# plt.show()

plot_results(log_dir, 'walltime_hrs')
plt.savefig(log_dir+"reward_vs_walltime_smoothed.png")
# plt.show()

plot_steps(log_dir)
plt.savefig(log_dir+"step_vs_timesteps_smoothed.png")

plot_steps_min(log_dir)
plt.savefig(log_dir+"stepmin_vs_timesteps_smoothed.png")
