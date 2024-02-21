# """
#     Serve per generare valutare parametri e iperparametri dei vari modelli, così da scegliere i migliori
#     Ci si aspetta che il modello sia già stato addestrato, il suo path va passato nell'argomento model_path
# """

# from env.custom_hopper import *
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3 import PPO
# import matplotlib.pyplot as plt
# from utils import curve_to_plot, train, test, test_plot
# import os
# import sys
# import argparse


# allowed_envs = [
#     "CustomHopper-v0",
#     "CustomHopper-source-v0",
#     "CustomHopper-source-udr-v0",
#     "CustomHopper-source-deception-v0",
#     "CustomHopper-target-v0",
# ]

# def parse_args(args=sys.argv[1:]):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, default="CustomHopper-source-deception-v0", help=f'{allowed_envs = }')
#     parser.add_argument("--train_timesteps", "-tt", type=int, default=20_000, help="Number of episodes to train for")
#     parser.add_argument("--model_path", "-mp", type=str, default=None, help="Path to the model to test")
#     # parser.add_argument("--render_training", action='store_true', help="Render each frame during training. Will be slower.")
#     # parser.add_argument("--render_test", action='store_true', help="Render test")
#     return parser.parse_args(args)

# # Compatibilità Enf
# import matplotlib as mpl
# mpl.use("GTK3Agg")

# def main(args):
#     curves=[]
#     seeds=[i for i in range(1,6)]
#     for s in seeds:
#         print("Learning with seed: ", s)
#         env = gym.make(args.env)
#         model, env = train(env, Seed=s, total_timesteps=args.train_timesteps)
#         x,y = curve_to_plot(f"./tmp/gym/{args.env}/")
#         curves.append(y)

#     # plot
#     fig = plt.figure("Learning curves for different seeds")
#     for i in range(len(seeds)):
#         plt.plot(curves[i])
#         plt.xlabel("Number of Timesteps")
#         plt.ylabel("Rewards")
#         plt.title(f"Learning curves for different seeds, env: {args.env}")
#         plt.legend([f"{seeds = }"])
#         plt.show()

#     # further inspecting of the learning rate

#     curves_lr=[]
#     learning_rates=[0.001, 0.003, 0.005]
#     for l in learning_rates:
#         print("learn with lr: ", l)
#         env = gym.make(args.env)
#         model, env = train(env, Seed=4, lr=l)
#         x,y = curve_to_plot(f"./tmp/gym/{args.env}/")
#         curves_lr.append(y)

#     # plot
#     fig = plt.figure("Learning curves for different learning rates")
#     for i in range(len(curves_lr)):
#         plt.plot(curves_lr[i])
#         plt.xlabel("Number of Timesteps")
#         plt.ylabel("Rewards")
#         plt.title("Learning curves for different learning rates")
#         plt.legend(f"{learning_rates = }")
#         plt.show()


# if __name__ == "__main__":
#     args = parse_args()

#     if not os.path.exists(str(args.model_path)):
#         exit("Model path not found. Exiting.")

#     if arg.

#     main(args)

import gym
import gym.spaces

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from env.custom_hopper import *
from env.adversarial_agent import *

from datetime import datetime

agent_env = gym.make('CustomHopper-source-deception-v0')

seeds = list(range(1,2))

for seed in seeds:
    print(f"Learning with {seed = }, {datetime.now()}")
    logdir = "./tmp/gym/adversarial_agent_seed" + str(seed) + "/"
    
    model = PPO('MlpPolicy', agent_env, seed)
    deception_env = gym.make("AdversarialAgent-v0",agent=model, agent_env=agent_env,num_ep=10)
    deception = SAC('MlpPolicy', deception_env)
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
        model.learn(500, progress_bar=True)

        total_episodes+=1
        print('\n',f'{total_episodes}','\n')

    model.save(f'deception_model_agent_dr_seed{seed}.mdl')