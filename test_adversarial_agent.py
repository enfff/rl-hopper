import sys
import argparse

print('This program parses arguments! Run with --help for more information')

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action='store_true', default=False, help="Render the environment during test; default: False")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of episodes used for testing; default: 50")
    parser.add_argument("--path", type=str, default="deception_model_agent_dr.mdl", help="Path to the model to be tested; default: deception_model_agent_dr.mdl")
    return parser.parse_args(args)

args = parse_args()
print(args)

from env.custom_hopper import *

import gym
import gym.spaces
from stable_baselines3.common.monitor import Monitor

from utils import load_model, test, test_plot

## Policy Evaluation

env_target = gym.make('CustomHopper-target-v0')
env_target_monitor = Monitor(env_target, "tmp/gym/aasource_target/")

# model = load_model('ppo', env_target, f'deception_model_agent_dr_seed{seed}.mdl')
model = load_model('ppo', env_target, args.path)
rew, lens = test(model, env_target_monitor, render=args.render, n_val_episodes=args.test_episodes)
test_plot(rew, lens, title=f"adversarial agent -> target")

env_target.close()