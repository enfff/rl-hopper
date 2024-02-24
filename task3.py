# TASK 3

"""
  Task 3  Train two agents with your algorithm of choice, on the source and target domains respectively. Then, test each model and
  report its average return over 50 test episodes. In particular, report results for the following “training→test” configurations: 
  source→source, 
  source→target (lower bound), 
  target→target (upper bound).
  Test with different hyperparameters and report the best results found together with the parameters used. 
"""

from utils import load_model
from datetime import datetime
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import os
from utils import train, test, test_plot

# Compatibilità Enf
import matplotlib as mpl
mpl.use("GTK3Agg")

start_time = datetime.now()

if not os.path.exists("target_model.zip"):
    print("Training target")
    env = gym.make('CustomHopper-target-v0')
    model, env = train(env, total_timesteps=100_000)
    model.save("./target_model")
    print(f"program ran for: {datetime.now() - start_time} ")


env = gym.make('CustomHopper-target-v0')
model = load_model('ppo', env, 'target_model')
rew, lens = test(model, Monitor(env, "./tmp/gym/target_model/"))
test_plot(rew, lens, title="target")

# Testing
# load source model from the previous task
model_src = PPO.load("source_model")
# policy_src = model_src.policy

model_trg = PPO.load("target_model")
# policy_trg = model_trg.policy

source = gym.make('CustomHopper-source-v0')
target = gym.make('CustomHopper-target-v0')

# # source -> source              # We don't really need this
# monitor_src= Monitor(source)
# rew1,lens1 = test(policy_src, monitor_src, False)
# source.close()
# print("test source -> source")
# # source.render()
# test_plot(rew1,lens1, title="source -> source")

# source -> target
monitor_trg = Monitor(target)
rew2, lens2 = test(model_src, monitor_trg, False)
print("test source -> target")
test_plot(rew2, lens2, title="source -> target")

# target -> target
monitor_trg2 = Monitor(target)
rew3, lens3 = test(model_trg, monitor_trg2, False)
target.close()
print("test target -> target")
test_plot(rew3, lens3, title="target -> target")
