# import gym
# from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Soluzione tornaconti Enf
import matplotlib as mpl
mpl.use("GTK3Agg")


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def curve_to_plot(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    ''' fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show() '''
    return x, y


def train(env, Seed=None, lr=0.003, total_timesteps=200000):

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    # masses of each link of the Hopper
    print('Dynamics parameters:', env.get_parameters())

    env = Monitor(env, "./")

    model = PPO("MlpPolicy", env, learning_rate=lr, device='cpu', seed=Seed)
    trained_mdl = model.learn(total_timesteps=total_timesteps)
    return trained_mdl, env


def test(model, env, render=False):

    rew, lens = evaluate_policy(
        model, env, n_eval_episodes=100, return_episode_rewards=True, render=render)
    return rew, lens


def test_plot(rew, lens, title=None):
    # average return over 50 episodes

    avg = []
    for i in range(len(rew)):
        if i > 50:
            avg.append(np.mean(rew[-50:]))
        else:
            avg.append(np.mean(rew))
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.plot(rew)
    plt.plot(avg, "r")
    plt.grid(True)
    plt.title(f"episode rewards {title}")
    plt.xlabel("episodes")
    plt.legend(['reward per episode', 'avg 50-episodes reward'])
    plt.subplot(212)
    plt.plot(lens)
    plt.grid(True)
    plt.title(f"episode lengths {title}")
    plt.xlabel("episodes")

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                        wspace=0.35)
    plt.show()


def create_model(alg, env, seed=None):
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, seed=seed)
    elif alg == 'sac':
        model = SAC("MlpPolicy", env, seed=seed)
    else:
        raise ValueError(f"RL Algo not supported: {alg}")
    return model


def load_model(alg, env, file):
    if alg == 'ppo':
        model = PPO.load(file, env=env)
    elif alg == 'sac':
        model = SAC.load(file, env=env)
    else:
        raise ValueError(f"RL Algo not supported: {alg}")
    return model


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.ffigigure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title)
    plt.show()
