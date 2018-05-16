

# 3rd party modules
import gym
import numpy as np
from gym import spaces

class MetaBanditEnv(gym.Env):


    def __init__(self):
        self.__version__ = "0.1.0"
        # print("MetaBandit - Version {}".format(self.__version__))

        # General variables defining the environment

        self.current_ep = 0

    def set_params(self, n_eps, n_levers):
        self.n_levers = n_levers
        self.n_eps = n_eps

        self.ps = np.random.rand(self.n_levers)

        # prev_action one-hot, prev_reward, eps_remaining
        self.state = np.zeros([self.n_levers + 2])
        self.state[-1] = self.n_eps

        self.action_space = spaces.Discrete(self.n_levers)

        self.observation_space = \
            spaces.Box(np.zeros([self.n_levers + 2]),
                       np.array([1. for _ in range(self.n_levers)] + [1., self.n_eps]),
                       dtype=np.float32)

    def step(self, action):
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

        # self.take_action(action)
        reward = 1. if np.random.rand() <  self.ps[action] else 0.
        ob = self.get_state()
        self.current_ep += 1
        self.prev_action_idx = action
        self.prev_reward = reward
        done = self.current_ep == self.n_eps
        info = {}
        return ob, reward, done, info

    # def take_action(self, action):
    #     return


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.ps = np.random.rand(self.n_levers)

        self.current_ep = 0

        self.prev_action_idx = None
        self.prev_reward = 0

        self.curr_episode = 0
        self.history = []
        return self.get_state()

    def render(self, mode='human', close=False):
        return

    def get_state(self):
        """Get the observation."""
        x = np.zeros(self.n_levers)
        if self.prev_action_idx:
            x[self.prev_action_idx] = 1
        else:
            pass
        ob = np.concatenate([x, [self.prev_reward, self.n_eps-self.current_ep]])
        return ob

    def seed(self, seed):
        np.random.seed(seed)



