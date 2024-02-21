# TASK 2

"""
  Task 2  Implement a reinforcement learning pipeline to train a simple control policy for the Hopper environment.
  To this end, you’ll make use of a third-party library to train an agent with state-of-the-art reinforcement learning
  algorithms such as PPO and SAC. In particular, follow the steps below, and make sure to go through the provided external resources:
  Create a script using the third-party library stable-baselines3 (sb3) and train the Hopper agent with one algorithm of choice between PPO [8] and SAC [7].
  Use the provided template in train.py as a starting point. It is okay to look at publicly available code for reference,
  but it’s likely easier and more helpful to study the sb3 documentation and understand how to implement the code by yourself.

"""

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

# train with different seeds
# curves=[]
# seeds=[i for i in range(1,6)]
# for s in seeds:
#   print("learn with seed: ",s)
#   env = gym.make('CustomHopper-source-v0')
#   model, env = train(env, Seed=s)
#   x,y=curve_to_plot("./")
#   curves.append(y)

# # plot
# fig = plt.figure("Learning curves for different seeds")
# for i in range(len(seeds)):
#   plt.plot(curves[i])
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning curves for different seeds")
# plt.legend(["1","2","3","4","5"])
# plt.show()

# # inspecting the learning rate

# curves_lr=[]
# lrs=[0.03, 0.003, 0.0003]
# for l in lrs:
#   print("learn with lr: ",l)
#   env = gym.make('CustomHopper-source-v0')
#   model, env = train(env, Seed=5, lr=l)
#   x,y=curve_to_plot("./")
#   curves_lr.append(y)

# # plot
# fig = plt.figure("Learning curves for different learning rates")
# for i in range(3):
#   plt.plot(curves_lr[i])
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning curves for different learning rates")
# plt.legend(['0.03', '0.003', '0.0003'])
# plt.show()

# further inspecting of the learning rate

# lrs=[0.001, 0.003, 0.005]
# for l in lrs:
#   print("learn with lr: ",l)
#   env = gym.make('CustomHopper-source-v0')
#   model, env = train(env, Seed=5, lr=l)
#   x,y=curve_to_plot("./")
#   curves_lr.append(y)

# # plot
# fig = plt.figure("Learning curves for different learning rates")
# for i in range(3,len(curves_lr)):
#   plt.plot(curves_lr[i])
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning curves for different learning rates")
# plt.legend(['0.001', '0.003', '0.005'])
# plt.show()

# BEST LEARNING RATE: 0.001

if not os.path.exists("source.zip"):
    env = gym.make('CustomHopper-source-v0')
    model, env = train(env, Seed=5, lr=0.001)
    rew, lens = test(model, env)
    test_plot(rew, lens, title="source")
    model.save("./source")
    exit()

env = gym.make('CustomHopper-source-v0')
model = load_model('ppo', env, 'source')
rew, lens = test(model, Monitor(env, "./tmp/gym/source/"))
test_plot(rew, lens, title="source")