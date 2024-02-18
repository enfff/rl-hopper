"""Test a random policy on the Gym Hopper environment


    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gym
from env.custom_hopper import *
env = gym.make('CustomHopper-source-v0')
# env = gym.make('CustomHopper-target-v0')

print('State space:', env.observation_space)  # state-space
print('Action space:', env.action_space)  # action-space
print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

## TASK 1

n_episodes = 5

for episode in range(n_episodes):
  done = False
  observation = env.reset()	 # Reset environment to initial state

  while not done:  # Until the episode is over

    action = env.action_space.sample()	# Sample random action

    observation, reward, done, info = env.step(action)	# Step the simulator to the next timestep


## TASK 2

import gym
from env.custom_hopper import *
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Soluzione tornaconti Enf
import matplotlib as mpl
mpl.use("GTK4Agg")
# assert(mpl.get_backend() == 'GTK4Agg')

import gym
from env.custom_hopper import *
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def curve_to_plot(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    ''' fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show() '''
    return x,y


def train(env, Seed=None, lr=0.003):

  print('State space:', env.observation_space)  # state-space
  print('Action space:', env.action_space)  # action-space
  print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

  env= Monitor(env, "./")

  model = PPO("MlpPolicy", env, learning_rate=lr, device='cpu', seed=Seed)
  trained_mdl = model.learn(total_timesteps=200000)
  return trained_mdl, env


def test(model,env):

  rew, lens= evaluate_policy(model, env, n_eval_episodes= 100, return_episode_rewards=True)
  return rew,lens


def test_plot(rew, lens):
  # average return over 50 episodes

  avg=[]
  for i in range(len(rew)):
    if i > 50:
      avg.append(np.mean(rew[-50:]))
    else:
      avg.append(np.mean(rew))

  plt.figure()
  plt.subplot(211)
  plt.plot(rew)
  plt.plot(avg,"r")
  plt.grid(True)
  plt.title("episode rewards")
  plt.xlabel("episodes")
  plt.legend(['reward per episode', 'avg 50-episodes reward'])
  plt.subplot(212)
  plt.plot(lens)
  plt.grid(True)
  plt.title("episode lengths")
  plt.xlabel("episodes")

  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                      wspace=0.35)
  plt.show()


# inspecting the seed

curves=[]
seeds=[i for i in range(1,6)]
for s in seeds:
  print("learn with seed: ",s)
  env = gym.make('CustomHopper-source-v0')
  model, env = train(env, Seed=s)
  x,y=curve_to_plot("./")
  curves.append(y)

# plot
fig = plt.figure("Learning curves for different seeds")
for i in range(len(seeds)):
  plt.plot(curves[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different seeds")
plt.legend(["1","2","3","4","5"])
plt.show()

# inspecting the learning rate

curves_lr=[]
lrs=[0.03, 0.003, 0.0003]
for l in lrs:
  print("learn with lr: ",l)
  env = gym.make('CustomHopper-source-v0')
  model, env = train(env, Seed=5, lr=l)
  x,y=curve_to_plot("./")
  curves_lr.append(y)

# plot
fig = plt.figure("Learning curves for different learning rates")
for i in range(3):
  plt.plot(curves_lr[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different learning rates")
plt.legend(['0.03', '0.003', '0.0003'])
plt.show()

# further inspecting of the learning rate

lrs=[0.001, 0.003, 0.005]
for l in lrs:
  print("learn with lr: ",l)
  env = gym.make('CustomHopper-source-v0')
  model, env = train(env, Seed=5, lr=l)
  x,y=curve_to_plot("./")
  curves_lr.append(y)

# plot
fig = plt.figure("Learning curves for different learning rates")
for i in range(3,len(curves_lr)):
  plt.plot(curves_lr[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different learning rates")
plt.legend(['0.001', '0.003', '0.005'])
plt.show()

# BEST LEARNING RATE: 0.001


env = gym.make('CustomHopper-source-v0')
model, env = train(env, Seed=5, lr=0.001)
rew,lens = test(model,env)
test_plot(rew,lens)

model.save("./source")

## TASK 3

# training policy on target domain

# inspecting the seed

curves=[]
seeds=[i for i in range(1,6)]
for s in seeds:
  print("learn with seed: ",s)
  env = gym.make('CustomHopper-target-v0')
  model, env = train(env, Seed=s)
  x,y=curve_to_plot("./")
  curves.append(y)

# plot
fig = plt.figure("Learning curves for different seeds")
for i in range(len(seeds)):
  plt.plot(curves[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different seeds")
plt.legend(["1","2","3","4","5"])
plt.show()

# inspecting the learning rate

curves_lr=[]
lrs=[0.03, 0.003, 0.0003]
for l in lrs:
  print("learn with lr: ",l)
  env = gym.make('CustomHopper-source-v0')
  model, env = train(env, Seed=1, lr=l)
  x,y=curve_to_plot("./")
  curves_lr.append(y)

# plot
fig = plt.figure("Learning curves for different learning rates")
for i in range(3):
  plt.plot(curves_lr[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different learning rates")
plt.legend(['0.03', '0.003', '0.0003'])
plt.show()

# further inspecting of the learning rate

lrs=[0.001, 0.003, 0.005]
for l in lrs:
  print("learn with lr: ",l)
  env = gym.make('CustomHopper-source-v0')
  model, env = train(env, Seed=1, lr=l)
  x,y=curve_to_plot("./")
  curves_lr.append(y)

# plot
fig = plt.figure("Learning curves for different learning rates")
for i in range(6,len(curves_lr)):
  plt.plot(curves_lr[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different learning rates")
plt.legend(['0.001', '0.003', '0.005'])
plt.show()

# BEST LEARNING RATE: 0.001

env = gym.make('CustomHopper-target-v0')
model, env = train(env, Seed=5, lr=0.001)
model.save("./target2")
rew,lens = test(model,env)
test_plot(rew,lens)

# load source model from the previous task
model_src= PPO.load("source")
policy_src= model_src.policy

model_trg= PPO.load("target")
policy_trg= model_trg.policy

source= gym.make('CustomHopper-source-v0')
target= gym.make('CustomHopper-target-v0')

# source -> source
monitor_src= Monitor(source)
rew1,lens1 = test(policy_src, monitor_src)
source.close()
print("test source -> source")
test_plot(rew1,lens1)

# source -> target
monitor_trg= Monitor(target)
rew2,lens2 = test(policy_src,monitor_trg)
print("test source -> target")
test_plot(rew2,lens2)

# target -> target
monitor_trg2= Monitor(target)
rew3,lens3 = test(policy_trg,monitor_trg2)
target.close()
print("test target -> target")
test_plot(rew3,lens3)


## TASK 4

from env.custom_hopper_UDR import *

curves=[]
seeds=[i for i in range(1,6)]
for s in seeds:
  print("learn with seed: ",s)
  env = gym.make('CustomHopperUDR-source-v0')
  model, env = train(env, Seed=s)
  x,y=curve_to_plot("./")
  curves.append(y)

# plot
fig = plt.figure("Learning curves for different seeds")
for i in range(len(seeds)):
  plt.plot(curves[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different seeds")
plt.legend(["1","2","3","4","5"])
plt.show()

# further inspecting of the learning rate

curves_lr=[]
lrs=[0.001, 0.003, 0.005]
for l in lrs:
  print("learn with lr: ",l)
  env = gym.make('CustomHopperUDR-source-v0')
  model, env = train(env, Seed=4, lr=l)
  x,y=curve_to_plot("./")
  curves_lr.append(y)

# plot
fig = plt.figure("Learning curves for different learning rates")
for i in range(len(curves_lr)):
  plt.plot(curves_lr[i])
plt.xlabel("Number of Timesteps")
plt.ylabel("Rewards")
plt.title("Learning curves for different learning rates")
plt.legend(['0.001', '0.003', '0.005'])
plt.show()

# BEST LEARNING RATE: 0.003

from env.custom_hopper_UDR import *

env = gym.make('CustomHopperUDR-source-v0')
model, env= train(env,Seed=4, lr= 0.003)
model.save("./source_udr")
rew,lens = test(model,env)
test_plot(rew,lens)

from env.custom_hopper_UDR import *

source_udr= PPO.load("source_udr")
policy_udr= source_udr.policy

print("test UDR source -> source")
source= gym.make('CustomHopper-source-v0')
src_monitor= Monitor(source,"./")
rew_src,lens_src = test(policy_udr,src_monitor)
test_plot(rew_src,lens_src)

print("test UDR source -> target")
target_UDR= gym.make('CustomHopper-target-v0')
trg_monitor= Monitor(target_UDR,"./")
rew_trg,lens_trg = test(policy_udr,trg_monitor)
test_plot(rew_trg,lens_trg)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import torch

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DeceptionNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        x = torch.abs(self.net(x))
        return x

#Setup the env, the deception net and the ppo model
env = gym.make('CustomHopper-source-v0')
device= torch.device("cuda")
deception_net = DeceptionNet(3,3)
deception_net = deception_net.to(device)
model = PPO('MlpPolicy', env)
#Number of times to train each network
batch_size = 10
total_episodes = 0
#Number of maximum step for episode
max_step = 500
#Set max episode to 1 via callback. This stops the training process after max_episodes_per_training episode
max_episodes_per_training = 1
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes_per_training)
#DeceptionNet Optimizer
lr = 1e-3
optimizer = torch.optim.Adam(deception_net.parameters(), lr=lr)
#Get current masses
masses = env.get_parameters()
masses_to_randomize = masses[1:]
masses_to_randomize = torch.from_numpy(masses_to_randomize).to(torch.float32)
rewards = []

model.learn(1000)

while total_episodes<100:
    # Train DeceptionNet
    deception_net.train()
    for episode in range(batch_size):
        optimizer.zero_grad()
        input= torch.from_numpy(np.array([10,10,10])).to(torch.float32).to(device)
        new_masses = deception_net(input)
        #Evaluate the agent policy on the newly generated masses
        #THE FOLLOWING METHOD (set_custom_parameters) IS TO BE IMPLEMENTED IN THE ENV
        print(new_masses)
       # env.set_custom_parameters(new_masses.detach().numpy())
        env.set_parameters(new_masses.cpu().detach().numpy())
        #Update loss
        mean_reward, std_reward = evaluate_policy(model,env,n_eval_episodes=10)
        rewards.append(mean_reward)
        reward = torch.tensor(mean_reward, requires_grad = True).to(torch.float32)
        reward= torch.tensor([F.sigmoid(reward)], requires_grad= True)
        print(reward)
        #loss = F.mse_loss(reward,torch.zeros(1))
        loss = F.binary_cross_entropy_with_logits(reward,torch.zeros(1))
        print(loss)
        #Update masses
        masses_to_randomize = new_masses
        #Compute gradient descent on DeceptionNet
        loss.backward()
        for name, param in deception_net.named_parameters():
          if param.requires_grad:
            print(name, param.data, param.grad)
        optimizer.step()


    # Train Agent
    #Generate new masses
    with torch.no_grad():
        new_masses = deception_net(masses_to_randomize)
    #Set the newly obtained masses
    env.set_parameters(new_masses.cpu())
    #Learn for one episode
    model.learn(max_step)
    #Update masses to randomize
    masses_to_randomize = new_masses

    total_episodes+=1
    optimizer.zero_grad()
    print('\n',total_episodes,'\n')

model.save('deception_model.mdl')
env.close()

plt.plot(rewards)
plt.show()