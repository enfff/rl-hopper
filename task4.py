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

from env.custom_hopper_UDR import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from utils import curve_to_plot, train, test, test_plot

# Compatibilità Enf
import matplotlib as mpl
mpl.use("GTK3Agg")

# curves=[]
# seeds=[i for i in range(1,6)]
# for s in seeds:
#   print("learn with seed: ",s)
#   env = gym.make('CustomHopperUDR-source-v0')
#   model, env = train(env, Seed=s)
#   x,y = curve_to_plot("./")
#   curves.append(y)

# # plot
# fig = plt.figure("Learning curves for different seeds")
# for i in range(len(seeds)):
#   plt.plot(curves[i])
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning curves for different seeds")
# plt.legend(["1","2","3","4","5"])
# plt.show()

# # further inspecting of the learning rate

# curves_lr=[]
# lrs=[0.001, 0.003, 0.005]
# for l in lrs:
#   print("learn with lr: ",l)
#   env = gym.make('CustomHopperUDR-source-v0')
#   model, env = train(env, Seed=4, lr=l)
#   x,y=curve_to_plot("./")
#   curves_lr.append(y)

# # plot
# fig = plt.figure("Learning curves for different learning rates")
# for i in range(len(curves_lr)):
#   plt.plot(curves_lr[i])
# plt.xlabel("Number of Timesteps")
# plt.ylabel("Rewards")
# plt.title("Learning curves for different learning rates")
# plt.legend(['0.001', '0.003', '0.005'])
# plt.show()

# # BEST LEARNING RATE: 0.003

# from env.custom_hopper_UDR import *

# env = gym.make('CustomHopperUDR-source-v0')
# model, env= train(env,Seed=4, lr= 0.003)
# model.save("./source_udr")
# rew,lens = test(model,env)
# test_plot(rew,lens)

from env.custom_hopper_UDR import *

source_udr= PPO.load("source_udr")
policy_udr= source_udr.policy

print("test UDR source -> source")
source= gym.make('CustomHopper-source-v0')
src_monitor= Monitor(source,"./")
rew_src,lens_src = test(policy_udr,src_monitor, True)
test_plot(rew_src,lens_src)

print("test UDR source -> target")
target_UDR= gym.make('CustomHopper-target-v0')
trg_monitor= Monitor(target_UDR,"./")
rew_trg,lens_trg = test(policy_udr,trg_monitor, True)
test_plot(rew_trg,lens_trg)

# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
# from stable_baselines3.common.evaluation import evaluate_policy
# import gym

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# class DeceptionNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, 32),
#             # nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             # nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, output_size),
#         )

#     def forward(self, x):
#         x = torch.abs(self.net(x))
#         return x

# #Setup the env, the deception net and the ppo model
# env = gym.make('CustomHopper-source-v0')
# device= torch.device("cuda")
# deception_net = DeceptionNet(3,3)
# deception_net = deception_net.to(device)
# model = PPO('MlpPolicy', env)
# #Number of times to train each network
# batch_size = 10
# total_episodes = 0
# #Number of maximum step for episode
# max_step = 500
# #Set max episode to 1 via callback. This stops the training process after max_episodes_per_training episode
# max_episodes_per_training = 1
# callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes_per_training)
# #DeceptionNet Optimizer
# lr = 1e-3
# optimizer = torch.optim.Adam(deception_net.parameters(), lr=lr)
# #Get current masses
# masses = env.get_parameters()
# masses_to_randomize = masses[1:]
# masses_to_randomize = torch.from_numpy(masses_to_randomize).to(torch.float32)
# rewards = []

# model.learn(1000)

# while total_episodes<100:
#     # Train DeceptionNet
#     deception_net.train()
#     for episode in range(batch_size):
#         optimizer.zero_grad()
#         input= torch.from_numpy(np.array([10,10,10])).to(torch.float32).to(device)
#         new_masses = deception_net(input)
#         #Evaluate the agent policy on the newly generated masses
#         #THE FOLLOWING METHOD (set_custom_parameters) IS TO BE IMPLEMENTED IN THE ENV
#         print(new_masses)
#        # env.set_custom_parameters(new_masses.detach().numpy())
#         env.set_parameters(new_masses.cpu().detach().numpy())
#         #Update loss
#         mean_reward, std_reward = evaluate_policy(model,env,n_eval_episodes=10)
#         rewards.append(mean_reward)
#         reward = torch.tensor(mean_reward, requires_grad = True).to(torch.float32)
#         reward= torch.tensor([F.sigmoid(reward)], requires_grad= True)
#         print(reward)
#         #loss = F.mse_loss(reward,torch.zeros(1))
#         loss = F.binary_cross_entropy_with_logits(reward,torch.zeros(1))
#         print(loss)
#         #Update masses
#         masses_to_randomize = new_masses
#         #Compute gradient descent on DeceptionNet
#         loss.backward()
#         for name, param in deception_net.named_parameters():
#           if param.requires_grad:
#             print(name, param.data, param.grad)
#         optimizer.step()


#     # Train Agent
#     #Generate new masses
#     with torch.no_grad():
#         new_masses = deception_net(masses_to_randomize)
#     #Set the newly obtained masses
#     env.set_parameters(new_masses.cpu())
#     #Learn for one episode
#     model.learn(max_step)
#     #Update masses to randomize
#     masses_to_randomize = new_masses

#     total_episodes+=1
#     optimizer.zero_grad()
#     print('\n',total_episodes,'\n')

# model.save('deception_model.mdl')
# env.render()
# env.close()

# plt.plot(rewards)
# plt.show()