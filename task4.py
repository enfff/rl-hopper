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
from utils import train, test, test_plot
import os

# Compatibilità Enf
import matplotlib as mpl
mpl.use("GTK3Agg")

from datetime import datetime

start_time = datetime.now()

if not os.path.exists("dr_model.zip"):
    env = gym.make('CustomHopper-source-dr-v0')
    model, env = train(env, total_timesteps=100_000)
    model.save("./dr_model")
    print(f"Model trained in: {datetime.now() - start_time}")

source = gym.make('CustomHopper-source-v0')
source_udr = PPO.load("dr_model")
policy_udr = source_udr.policy

# print("test UDR source")
# rew, lens = test(source_udr, source, render=False)
# test_plot(rew,lens, title="UDR source")

# print("test UDR source -> source")
# src_monitor = Monitor(source,"tmp/gym/udrsource_to_source/")
# rew_src,lens_src = test(policy_udr,src_monitor, render=False)
# test_plot(rew_src,lens_src, title="UDR source -> source")

print("test dr source -> target")
target = gym.make('CustomHopper-target-v0')
trg_monitor = Monitor(target,"tmp/gym/drmodel_target/")
rew_trg,lens_trg = test(policy_udr,trg_monitor, render=False)
test_plot(rew_trg,lens_trg, title="dr source -> target")