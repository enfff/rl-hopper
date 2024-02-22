# from stable_baselines3.common.monitor import Monitor
# from utils import load_model
# import gym
# from env.custom_hopper import *
# from stable_baselines3 import PPO
# import matplotlib.pyplot as plt
# import os

# from utils import curve_to_plot, train, test, test_plot

# # Compatibilità Enf
# import matplotlib as mpl
# mpl.use("GTK3Agg")

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

import sys
import argparse

print('This program parses arguments! Run with --help for more information')

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action='store_true', default=False, help="Render the environment during test; default: False")
    parser.add_argument("--test_episodes", type=int, default=50, help="Number of episodes used for testing; default: 50")
    return parser.parse_args(args)

args = parse_args()
print(args)

from env.custom_hopper import *

import gym
import gym.spaces

from utils import load_model, test, test_plot

## Policy Evaluation

test_episodes = 50

env_target = gym.make('CustomHopper-target-v0')

seeds = list(range(1, 4))

for seed in seeds:
    model = load_model('ppo', env_target, f'deception_model_agent_dr_seed{seed}.mdl')
    # mean_reward, std_reward = evaluate_policy(model,env_target,n_eval_episodes=test_episodes)
    rew, lens = test(model, env_target, render=args.render, n_val_episodes=test_episodes)
    # print(f"Test reward (avg +/- std): ({rew} +/- {lens}) - Num episodes: {test_episodes}")
    test_plot(rew, lens, title=f", random seed ({seed})")

env_target.close()