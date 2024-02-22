## TASK 4

"""
Implement Uniform Domain Randomization (UDR) for the link masses of the Hopper robot.
In this setting, UDR refers to manually designing a uniform distribution over the three remaining masses in the source
environment (considering that the torso mass is fixed at -1 kg w.r.t. the target one) and performing training with values
that vary at each episode (sampled appropriately from the chosen distributions).
The underlying idea is to force the agent to maximize its reward and solve the task for a range of multiple environments
at the same time, such that its learned behavior may be robust to dynamics variations.
Note that, since the choice of the distribution is a hyperparameter of the method, the student has to manually try different
distributions in order to expect good results on the target environment.

Task 4  Train a UDR agent on the source environment with the same RL algorithm previously used. Later test the policy
obtained on both the source and target environments
"""

from env.custom_hopper import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from utils import curve_to_plot, train, test, test_plot
import os

# Compatibilità Enf
import matplotlib as mpl
mpl.use("GTK3Agg")

# curves=[]
# seeds=[i for i in range(1,6)]
# for s in seeds:
#   print("learn with seed: ",s)
#   env = gym.make('CustomHopperUDR-source-v0')
#   model, env = train(env, Seed=s)
#   x,y = curve_to_plot("./")
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

# # further inspecting of the learning rate

# curves_lr=[]
# lrs=[0.001, 0.003, 0.005]
# for l in lrs:
#   print("learn with lr: ",l)
#   env = gym.make('CustomHopperUDR-source-v0')
#   model, env = train(env, Seed=4, lr=l)
#   x,y=curve_to_plot("./")
#   curves_lr.append(y)

# # plot
# fig = plt.figure("Learning curves for different learning rates")
# for i in range(len(curves_lr)):
#   plt.plot(curves_lr[i])
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning curves for different learning rates")
# plt.legend(['0.001', '0.003', '0.005'])
# plt.show()

# # BEST LEARNING RATE: 0.003

from datetime import datetime

if not os.path.exists("source_udr.zip"):
    start_time = datetime.now()
    env = gym.make('CustomHopper-source-udr-v0')
    model, env = train(env, Seed = 4, lr = 0.003, total_timesteps=200000)
    model.save("./source_udr")
    print(f"Model trained in: {datetime.now() - start_time}")
    rew,lens = test(model,env)
    test_plot(rew,lens, title="UDR source")

source = gym.make('CustomHopper-source-v0')
source_udr = PPO.load("source_udr")
policy_udr = source_udr.policy

# print("test UDR source")
# rew, lens = test(source_udr, source, render=False)
# test_plot(rew,lens, title="UDR source")

# print("test UDR source -> source")
# src_monitor = Monitor(source,"tmp/gym/udrsource_to_source/")
# rew_src,lens_src = test(policy_udr,src_monitor, render=False)
# test_plot(rew_src,lens_src, title="UDR source -> source")

# print("test UDR source -> target")
# target = gym.make('CustomHopper-target-v0')
# trg_monitor = Monitor(target,"tmp/gym/udrsource_to_target/")
# rew_trg,lens_trg = test(policy_udr,trg_monitor, render=False)
# test_plot(rew_trg,lens_trg, title="UDR source -> target")