import gym
from env.custom_hopper import *

env_src = gym.make('CustomHopper-source-v0')
env_target = gym.make('CustomHopper-target-v0')

SEED = 123

print('State space:', env_src.observation_space)  # state-space
print('Action space:', env_src.action_space)  # action-space
print('Dynamics parameters:', env_src.get_parameters())  # masses of each link of the Hopper

print(f'Bodies defined in the environment:\n {env_src.sim.model.body_names}\n')
print(f'Mass of all the corresponding bodies:\n {env_src.sim.model.body_mass}\n')
print(f'Number of degrees of freedom (DoFs) of the robot:\n {env_src.sim.model.nv}\n')
print(f'Number of DoFs for each body:\n {env_src.sim.model.body_dofnum}\n')
print(f'Number of actuators:\n {env_src.sim.model.nu}\n')
#Box -> Contiuos action and state spaces
print(f'State space:\n {env_src.observation_space}\n')
print(f'Action space:\n {env_src.action_space}\n')
print(f'Mass of all the bodies of the target:\n {env_target.sim.model.body_mass}\n')

## Imports and global variables definition

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from utils import moving_average, create_model, load_model, plot_results

alg = 'ppo'
train_episodes = 100000
test_episodes = 50

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import gym.spaces
import numpy as np

import csv
import os

class AdversarialAgent(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,agent,agent_env,num_ep):

    # The action space is the masses of the hopper (),
    self.action_space = spaces.Box(
        low = np.array([1.0,1.0,1.0]),
        high = np.array([10.0, 10.0, 10.0])
    )
    
    # The observation is the reward of the agent trained with the
    # generated masses
    self.observation_space = spaces.Box(
        low = np.array([1.0,1.0,1.0]),
        high = np.array([10.0, 10.0, 10.0])
    )
    # masses of the hopper, reward, lunghezza media degli episodi

    # agent to use to generate the reward
    # to update at the end of every cycle
    self.agent = agent
    self.agent_env = agent_env

    #Training parameters
    # - number of episode on which to test the new masses
    self.num_ep = num_ep

  def step(self, action):
    """
    This method is the primary interface between environment and agent.
    """
    generated_masses = action
    returns = []
    #self.agent_env.set_custom_parameters(action)
    for t in range(self.num_ep):
      current_return = 0
      obs = self.agent_env.reset()
      self.agent_env.set_custom_parameters(generated_masses)
      for _ in range(500):
        agent_action, _ = self.agent.predict(obs)
        obs, reward, done, _ = self.agent_env.step(agent_action)
        current_return+=reward
        if done:
          break
      returns.append(current_return)
    #mean_reward, std_reward = evaluate_policy(self.agent,self.agent_env,n_eval_episodes=self.num_ep)
    mean_reward = np.mean(np.array(returns))

    reward = 1/mean_reward
    print(f'{generated_masses}->{mean_reward}->{reward}')

    return np.array([generated_masses],dtype=np.float32), reward, True, {}

  def reset(self):
    """
    This method resets the environment to its initial values.

    Returns:
        observation:    array
                        the initial state of the environment
    """
    #mean_reward, std_reward = evaluate_policy(self.agent,self.agent_env,n_eval_episodes=self.num_ep)
    masses = self.agent_env.get_parameters()
    masses = masses[1:]
    #print(masses)
    return np.array([masses],dtype=np.float32)

  def render(self, mode='human', close=False):
    """
    This methods provides the option to render the environment's behavior to a window
    which should be readable to the human eye if mode is set to 'human'.
    """
    pass

  def close(self):
    """
    This method provides the user with the option to perform any necessary cleanup.
    """
    pass

gym.envs.register(
        id="AdversarialAgent-v0",
        entry_point="%s:AdversarialAgent" % __name__,
        max_episode_steps=10
)

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3 import SAC, PPO

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
mean_reward, std_reward = evaluate_policy(model,env_target,n_eval_episodes=test_episodes)
print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {test_episodes}")
env_target.close()