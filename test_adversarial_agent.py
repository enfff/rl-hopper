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


from env.custom_hopper import *

import gym
import gym.spaces

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

## Policy Evaluation

train_episodes = 100000
test_episodes = 50

def load_model(alg, env, file):
    if alg == 'ppo':
        model = PPO.load(file, env=env)
    elif alg == 'sac':
        model = SAC.load(file, env=env)
    else:
        raise ValueError(f"RL Algo not supported: {alg}")
    return model

env_target = gym.make('CustomHopper-target-v0')
model = load_model('ppo', env_target, 'deception_model_agents/deception_model_agent_dr_seed1.mdl')
mean_reward, std_reward = evaluate_policy(model,env_target,n_eval_episodes=test_episodes)
print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {test_episodes}")
env_target.close()