import sys
import argparse
import os

print('This program parses arguments! Run with --help for more information')

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action='store_true', default=False, help="Render the environment during test; default: False")
    parser.add_argument("--test_episodes", type=int, default=50, help="Number of episodes used for testing; default: 50")
    parser.add_argument("--source_path", type=str, default="source_model.mdl", help="default: source_model.mdl")
    parser.add_argument("--target_path", type=str, default="target_model.mdl", help="default: target_model.mdl")
    parser.add_argument("--dr_path", type=str, default="dr_model.mdl", help="default: dr_model.mdl")
    parser.add_argument("--deception_path", type=str, default="deception_model_agent_dr.mdl", help="default: deception_model_agent_dr.mdl")
    return parser.parse_args(args)

args = parse_args()

print(args)

assert os.path.exists(args.source_path), "Source model file not found"
assert os.path.exists(args.target_path), "Target model file not found"
assert os.path.exists(args.deception_path), "Deception model file not found"
assert os.path.exists(args.dr_path), "Domain Randomization model file not found"

from env.custom_hopper import *

import gym
import gym.spaces
from stable_baselines3.common.monitor import Monitor

from utils import load_model, test, test_plot

## Policy Evaluation

env_target = gym.make('CustomHopper-target-v0')

env_target_monitor_source = Monitor(env_target, "tmp/gym/source_target/")
env_target_monitor_target = Monitor(env_target, "tmp/gym/target_target/")
env_target_monitor_dr = Monitor(env_target, "tmp/gym/drsource_target/")
env_target_monitor_deception = Monitor(env_target, "tmp/gym/deceptor_source_target/")

source_model = load_model('ppo', env_target_monitor_source, args.source_path)
target_model = load_model('ppo', env_target, args.target_path)
dr_model = load_model('ppo', env_target_monitor_dr, args.dr_path)
deception_model = load_model('ppo', env_target_monitor_deception, args.deception_path)

rew, lens = test(source_model, env_target_monitor_source, render=args.render, n_val_episodes=args.test_episodes)
test_plot(rew, lens, title=f"source -> target", save_filename="source_target")

rew, lens = test(target_model, env_target_monitor_target, render=args.render, n_val_episodes=args.test_episodes)
test_plot(rew, lens, title=f"target -> target", save_filename="target_target")

rew, lens = test(dr_model, env_target_monitor_dr, render=args.render, n_val_episodes=args.test_episodes)
test_plot(rew, lens, title=f"dr source -> target", save_filename="drsource_target")

rew, lens = test(deception_model, env_target_monitor_deception, render=args.render, n_val_episodes=args.test_episodes)
test_plot(rew, lens, title=f"deceptor source -> target", save_filename="deceptorsource_target")

env_target.close()