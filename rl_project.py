import gym
import gym.spaces

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from env.custom_hopper import *
from env.adversarial_agent import *

SEED = 4
env_target = gym.make('CustomHopper-target-v0')

# env_src = gym.make('CustomHopper-source-v0')
# print('State space:', env_src.observation_space)  # state-space
# print('Action space:', env_src.action_space)  # action-space
# print('Dynamics parameters:', env_src.get_parameters())  # masses of each link of the Hopper

# print(f'Bodies defined in the environment:\n {env_src.sim.model.body_names}\n')
# print(f'Mass of all the corresponding bodies:\n {env_src.sim.model.body_mass}\n')
# print(f'Number of degrees of freedom (DoFs) of the robot:\n {env_src.sim.model.nv}\n')
# print(f'Number of DoFs for each body:\n {env_src.sim.model.body_dofnum}\n')
# print(f'Number of actuators:\n {env_src.sim.model.nu}\n')

# #Box -> Continuos action and state spaces
# print(f'State space:\n {env_src.observation_space}\n')
# print(f'Action space:\n {env_src.action_space}\n')
# print(f'Mass of all the bodies of the target:\n {env_target.sim.model.body_mass}\n')

## Imports and global variables definition

import os
from utils import load_model

agent_env = gym.make('CustomHopper-deception-v0')
aa_log_dir = "./tmp/gym/adversarial_agent/"
os.makedirs(aa_log_dir, exist_ok=True)
agent_env = Monitor(agent_env, aa_log_dir)
model = PPO('MlpPolicy', agent_env, seed=SEED)

deception_env = gym.make("AdversarialAgent-v0",agent=model,agent_env=agent_env,num_ep=10)
de_log_dir = "./tmp/gym/deception_environment/"
os.makedirs(de_log_dir, exist_ok=True)
deception_env = Monitor(deception_env, de_log_dir)

#check_env(deception_env)

# Train model (if not exists)
if not os.path.exists('deception_model_agent_dr.mdl'):
    deception = SAC('MlpPolicy', deception_env) # deception uses random seed
    agent_env.set_deceptor(deception)

    total_episodes = 0
    max_step = 500

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=10)

    thighs = []
    legs = []
    foots = []

    while total_episodes<200:

        #Train deception module
        for i in range(20):
            deception.learn(1, callback_max_episodes)

        #Train Agent
        model.learn(500)

        total_episodes+=1
        print('\n',f'{total_episodes}','\n')

    model.save('deception_model_agent_dr.mdl')

## Policy Evaluation

env_target = gym.make('CustomHopper-target-v0')
env_target = Monitor(env_target, "./tmp/gym/target/")
model = load_model('ppo', env_target, 'deception_model_agent_dr.mdl')
mean_reward, std_reward = evaluate_policy(model,env_target,n_eval_episodes=test_episodes, render=True)
print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {test_episodes}")
env_target.close()