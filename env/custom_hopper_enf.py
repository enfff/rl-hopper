"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, domain_randomization_type=None):
        """
            domain:     "source",
                        "target"
            dr_type:    "udr"       -> (uniform domain randomization) set_random_parameters,
                        "deception" -> _generate_parameters,
                        None
        """
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        self.domain_randomization_type = domain_randomization_type

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0


    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        thigh = self.original_masses[1]
        leg = self.original_masses[2]
        foot = self.original_masses[3]

        thigh = np.random.uniform(thigh-thigh/2,thigh+thigh/2)
        leg = np.random.uniform(leg-leg/2,leg+leg/2)
        foot = np.random.uniform(foot-foot/2,foot+foot/2)

        # if self.domain_randomization_type == "deception":
        #     # No fucking why, but it avoids a bug
        #     return np.array([self.original_masses[0], thigh, leg, foot])

        return np.array([self.sim.model.body_mass[1], thigh, leg, foot])

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def set_custom_parameters(self, params):
        self.sim.model.body_mass[2:] = params

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,    action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def set_deceptor(self, deceptor):
      self.deceptor = deceptor

    def _generate_parameters(self):
      masses = self.original_masses[1:]
      obs = np.array([*masses],dtype=np.float32)
      action, _ = self.deceptor.predict(obs)
      self.sim.model.body_mass[2:] = action

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)

        if self.domain_randomization_type == "udr":
          self.set_random_parameters()
        elif self.domain_randomization_type == "deception":
          self._generate_parameters()
        
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="CustomHopper-source-udr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "domain_randomization_type": "udr"}
)

gym.envs.register(
        id="CustomHopper-source-deception-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "domain_randomization_type": "deception"}
)