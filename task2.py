# TASK 2

"""
  Task 2  Implement a reinforcement learning pipeline to train a simple control policy for the Hopper environment.
  To this end, you’ll make use of a third-party library to train an agent with state-of-the-art reinforcement learning
  algorithms such as PPO and SAC. In particular, follow the steps below, and make sure to go through the provided external resources:
  Create a script using the third-party library stable-baselines3 (sb3) and train the Hopper agent with one algorithm of choice between PPO [8] and SAC [7].
  Use the provided template in train.py as a starting point. It is okay to look at publicly available code for reference,
  but it’s likely easier and more helpful to study the sb3 documentation and understand how to implement the code by yourself.

"""

from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from utils import load_model
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os

from utils import curve_to_plot, train, test, test_plot

# Compatibilità Enf
import matplotlib as mpl
mpl.use("GTK3Agg")


start_time = datetime.now()

# Train if not present
if not os.path.exists("source_model.zip"):
    print("Training started for source")
    env = gym.make('CustomHopper-source-v0')
    model, env = train(env, total_timesteps=100_000)
    # rew, lens = test(model, env)
    # test_plot(rew, lens, title="source")
    print(f"program ran for: {datetime.now() - start_time} ")
    model.save("./source_model")

env = gym.make('CustomHopper-source-v0')
model = load_model('ppo', env, 'source_model')
rew, lens = test(model, Monitor(env, "./tmp/gym/source_model/"))
test_plot(rew, lens, title="source")