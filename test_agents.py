import sys
import argparse

print('This program parses arguments! Run with --help for more information')

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action='store_true', default=False, help="Render the environment during test; default: False")
    parser.add_argument("--test_episodes", type=int, default=100, help="Number of episodes used for testing; default: 50")
    parser.add_argument("--model_path", type=str, default="deception_model_agent_dr.mdl", help="Path to the model to be tested; default: deception_model_agent_dr.mdl")
    # parser.add_argument("--env", type=str, default="target", help="Which environment to use: source or target; default: source")
    return parser.parse_args(args)


args = parse_args()
# if args.env not in ["source", "target"]:
#     raise ValueError("env must be one of 'source' or 'target'")
print(args)

from env.custom_hopper import *

import gym
import gym.spaces
from stable_baselines3.common.monitor import Monitor
import os
from pathlib import Path

from utils import load_model, test, test_plot

assert(os.path.exists(args.model_path))

model_name = Path(args.model_path).stem

if model_name.contains("udr_source"):
    model_name = "udr_source"
if model_name.contains("source"):
    model_name = "source"
elif model_name.contains("target"):
    model_name = "target"
elif model_name.contains("deception") or model_name.contains("adversarial") or model_name.contains("dr"):
    model_name = "aa source"



print(f"{model_name = }")

env_target = gym.make(f'CustomHopper-{model_name}-v0')
env_target_monitor = Monitor(env_target, f"tmp/gym/{env_target}_{model_name}/")

model = load_model('ppo', env_target, args.model_path)
rew, lens = test(model, env_target_monitor, render=args.render, n_val_episodes=args.test_episodes)
test_plot(rew, lens, title=f"{model_name} -> target") # 

env_target.close()



# import sys
# import argparse

# print('This program parses arguments! Run with --help for more information')

# def parse_args(args=sys.argv[1:]):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--render", action='store_true', default=False, help="Render the environment during test; default: False")
#     parser.add_argument("--test_episodes", type=int, default=100, help="Number of episodes used for testing; default: 50")
#     parser.add_argument("--model_path", type=str, default="deception_model_agent_dr.mdl", help="Path to the model to be tested; default: deception_model_agent_dr.mdl")
#     # parser.add_argument("--env", type=str, default="target", help="Which environment to use: source or target; default: source")
#     return parser.parse_args(args)


# args = parse_args()
# # if args.env not in ["source", "target"]:
# #     raise ValueError("env must be one of 'source' or 'target'")
# print(args)

# from env.custom_hopper import *

# import gym
# import gym.spaces
# from stable_baselines3.common.monitor import Monitor
# import os
# from pathlib import Path

# from utils import load_model, test, test_plot

# assert(os.path.exists(args.model_path))

# model_name = "deception-source"


# # env_target = gym.make(f'CustomHopper-{model_name}-v0')
# # env_target_monitor = Monitor(env_target, f"tmp/gym/{model_name}_{env_target}/")

# # deception_env = gym.make("AdversarialAgent-v0",agent=model,agent_env=agent_env,num_ep=10)
# # deception = SAC('MlpPolicy', deception_env)

# env_target = gym.make('CustomHopper-target-v0')
# model = load_model('ppo', env_target, args.model_path)

# rew, lens = test(model, env_target, render=args.render, n_val_episodes=args.test_episodes)
# test_plot(rew, lens, title=f"{model_name} -> target") # 

# env_target.close()