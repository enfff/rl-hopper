import gym
from gym import spaces
import numpy as np

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

    # masses of the hopper, reward, lunghezza media degli episodi
    self.observation_space = spaces.Box(
        low = np.array([1.0,1.0,1.0]),
        high = np.array([10.0, 10.0, 10.0])
    )

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
    # TODO add csv logging
    print(f'{generated_masses}->{mean_reward}->{reward}')

    return np.concatenate([np.array([generated_masses],dtype=np.float32)]), reward, True, {}

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
    # return np.array([masses],dtype=np.float32)
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