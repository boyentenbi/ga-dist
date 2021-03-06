

# 3rd party modules
import gym
import numpy as np
from gym import spaces

class MetaBanditEnv(gym.Env):


    def __init__(self):
        self.__version__ = "0.1.0"
        print("MetaBanditEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.n_levers = 2
        self.n_eps = 10
        self.ps = np.random.rand(self.n_levers)

        self.current_ep = 0
        self.history = []
        self.prev_action_idx = None
        self.prev_reward = None
        self.action_space = spaces.Discrete(self.n_levers)

        self.observation_space = spaces.Tuple((spaces.Discrete(self.n_levers), # Previous action taken
                                               spaces.Box(np.array([0, 0 ]), np.array([1., self.n_eps])), # Previous reward, eps remaining
                                               ))

    def _step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        self._take_action(action)
        reward = 1. if np.random.rand() <  self.ps[action] else 0.
        ob = self._get_state()
        self.current_ep += 1
        done = self.current_ep == self.n_eps
        info = {}
        return ob, reward, done, info


    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.ps = np.random.rand(self.n_levers)

        self.current_ep = 0

        self.prev_action_idx = None
        self.prev_reward = None

        self.curr_episode = 0
        self.history = []
        return self._get_state()

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        x = np.zeros(self.n_levers)
        x[self.prev_action_idx] = 1.
        ob = (x, [self.prev_reward, self.n_eps-self.current_ep],)
        return ob



