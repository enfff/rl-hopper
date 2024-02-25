from env.custom_hopper import *

import gym
from gym import spaces
import gym.spaces
import numpy as np
import sys
import argparse

print('This program parses arguments! Run with --help for more information')

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--render", action='store_false', help="Render the environment during training")
    parser.add_argument("--save_mass_log", action='store_true', default=False, help="Save mass log during training in tmp/gym/modelname/mass_monitor.csv; default: False, slows training down significantly")
    parser.add_argument("--print_extrainfo", action='store_false', default=True, help="Disables printing the generated masses, mean_reward and reward during training; default: False")
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to train the adversarial agent (SAC); default: 200")
    parser.add_argument("--total_timesteps", type=int, default=500, help="Number of timesteps to train the adversarial agent (SAC); default: 500")
    parser.add_argument("--deception_train_steps", type=int, default=50, help="How many training steps the deceptor does for each episode; default: 50")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for all the models; default: 0.0003")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the training; default: None")
    return parser.parse_args(args)

args = parse_args()

assert(type(args.lr) == float), "Learning rate must be a float number"

print(args)


class AdversarialAgent(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,agent,agent_env,num_ep,logdir=None):

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

    #Training parameters
    # - number of episode on which to test the new masses
    self.num_ep = num_ep

    if args.save_mass_log:
      self.mass_monitor_filepath = f"{self.logdir}/mass_monitor.csv"
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
    
    if args.print_extrainfo:
      m1, m2, m3 = float(generated_masses[0]), float(generated_masses[1]), float(generated_masses[2])
      print(f'{float(m1):>10.4f}{float(m2):>10.4f}{float(m3):>10.4f}{float(mean_reward):^30.4f}{float(reward):<30.4f}')

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

    if args.save_mass_log:
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


from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3 import SAC, PPO
from datetime import datetime
import os
import csv

start_time = datetime.now()

if args.save_mass_log:
  logdir = f'tmp/gym/adversarial_agent/'
  mass_monitor_filepath = logdir + 'mass_monitor.csv'
  if not os.path.exists(logdir):
      os.makedirs(logdir, exist_ok=True)
else:
  logdir = None

if args.seed: print(f"Learning with {args.seed = }")

print(f"start time: {datetime.now()}")

agent_env = gym.make('CustomHopper-deception-v0')
model = PPO('MlpPolicy', agent_env, seed=args.seed, learning_rate=args.lr)

deception_env = gym.make("AdversarialAgent-v0", agent=model, agent_env=agent_env, num_ep=10, logdir=logdir)
#check_env(deception_env)

deception = SAC('MlpPolicy', deception_env, seed=args.seed, learning_rate=args.lr)

agent_env.set_deceptor(deception)

total_episodes = 0
max_step = 500

callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=10)

thighs = []
legs = []
foots = []

if args.print_extrainfo:
  info1 = "generated_masses"
  info2 = "mean_reward"
  info3 = "reward"
  print(f'{info1:^30} {info2:^25} {info3:^10} ')
  del info1, info2, info3 # Quick and dirty

while total_episodes < args.num_episodes:

    #Train deception module
    for i in range(args.deception_train_steps):
        deception.learn(1, callback_max_episodes)

    #Train Agent
    model.learn(args.total_timesteps)

    total_episodes+=1

    if args.print_extrainfo:
      print('\n',f'{total_episodes}','\n')

model.save(f'deception_model_agent_dr.mdl')

print(f"End of learning, program ran for: {datetime.now() - start_time} ")