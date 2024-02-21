from env.custom_hopper import *

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import gym.spaces
import numpy as np

import csv

class AdversarialAgent(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,agent,agent_env,num_ep,logdir):

    self.action_space = spaces.Box(
        low=np.array([1.0,1.0,1.0]),
        high=np.array([10.0, 10.0, 10.0])
    )
    # The observation is the reward of the agent trained with the
    # generated masses
    self.observation_space = spaces.Box(
        low = np.array([1.0,1.0,1.0]),
        high=np.array([10.0, 10.0, 10.0])
    )

    #agent to use to generate the reward
    #to update at the end of every cycle
    self.agent = agent
    self.agent_env = agent_env
    self.logdir = logdir
    self.mass_monitor_filepath = f"{self.logdir}/mass_monitor.csv"

    #Training parameters
    # - number of episode on which to test the new masses
    self.num_ep = num_ep

    with open(self.mass_monitor_filepath, 'w') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['mass1','mass2','mass3'])


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

    with open(self.mass_monitor_filepath, 'a') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(masses)

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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, PPO

from datetime import datetime

seeds = list(range(1,6))

import os

for seed in seeds:

    logdir = f'tmp/gym/adversarial_agent_seed{seed}/'
    mass_monitor_filepath = logdir + 'mass_monitor.csv'

    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    print(f"Learning with {seed = }, {datetime.now()}")
    agent_env = gym.make('CustomHopper-source-deception-v0')
    model = PPO('MlpPolicy', agent_env)

    deception_env = gym.make("AdversarialAgent-v0",agent=model,agent_env=agent_env,num_ep=10, logdir=logdir)
    #check_env(deception_env)

    deception = SAC('MlpPolicy', deception_env)

    agent_env.set_deceptor(deception)

    total_episodes = 0
    max_step = 500

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=10)

    thighs = []
    legs = []
    foots = []

    while total_episodes < 200:

        #Train deception module
        for i in range(200):
            deception.learn(1, callback_max_episodes)

        #Train Agent
        model.learn(500)

        total_episodes+=1
        print('\n',f'{total_episodes}','\n')

    model.save(f'deception_model_agent_dr_seed{seed}.mdl')

print(f"End of learning, 5 models generated, {datetime.now()}")