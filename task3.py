## TASK 3

"""
  Task 3  Train two agents with your algorithm of choice, on the source and target domains respectively. Then, test each model and
  report its average return over 50 test episodes. In particular, report results for the following “training→test” configurations: 
  source→source, 
  source→target (lower bound), 
  target→target (upper bound).
  Test with different hyperparameters and report the best results found together with the parameters used. 
"""

import gym
from env.custom_hopper import *
from stable_baselines3 import PPO,SAC
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# import matplotlib.pyplot as plt
# from stable_baselines3.common.results_plotter import load_results, ts2xy

import os
from utils import curve_to_plot, train, test, test_plot

# Compatibilità Enf
import matplotlib as mpl
mpl.use("GTK3Agg")

# training policy on target domain

# inspecting the seed

# curves=[]
# seeds=[i for i in range(1,6)]
# for s in seeds:
#   print("learn with seed: ",s)
#   env = gym.make('CustomHopper-target-v0')
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
#   model, env = train(env, Seed=1, lr=l)
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

# # further inspecting of the learning rate

# lrs=[0.001, 0.003, 0.005]
# for l in lrs:
#   print("learn with lr: ",l)
#   env = gym.make('CustomHopper-source-v0')
#   model, env = train(env, Seed=1, lr=l)
#   x,y=curve_to_plot("./")
#   curves_lr.append(y)

# # plot
# fig = plt.figure("Learning curves for different learning rates")
# for i in range(6,len(curves_lr)):
#   plt.plot(curves_lr[i])
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning curves for different learning rates")
# plt.legend(['0.001', '0.003', '0.005'])
# plt.show()

# # BEST LEARNING RATE: 0.001

if not os.path.exists("target.zip"):
  print("Training target")
  env = gym.make('CustomHopper-target-v0')
  model, env = train(env, Seed=5, lr=0.001)
  model.save("./target")
  rew,lens = test(model,env)
  test_plot(rew,lens)

from utils import load_model

env = gym.make('CustomHopper-target-v0')
model = load_model('ppo', env, 'target')
rew, lens = test(model, Monitor(env, "./tmp/gym/target/"))
test_plot(rew, lens, title="target")

## Testing
# load source model from the previous task
model_src = PPO.load("source")
policy_src = model_src.policy

model_trg= PPO.load("target")
policy_trg= model_trg.policy

source = gym.make('CustomHopper-source-v0')
target = gym.make('CustomHopper-target-v0')

# source -> source
monitor_src= Monitor(source)
rew1,lens1 = test(policy_src, monitor_src, False)
source.close()
print("test source -> source")
# source.render()
test_plot(rew1,lens1, title="source -> source")

# source -> target
monitor_trg= Monitor(target)
rew2,lens2 = test(policy_src,monitor_trg, False)
print("test source -> target")
test_plot(rew2,lens2, title="source -> target")

# target -> target
monitor_trg2= Monitor(target)
rew3,lens3 = test(policy_trg,monitor_trg2, False)
target.close()
print("test target -> target")
test_plot(rew3,lens3, title="target -> target")