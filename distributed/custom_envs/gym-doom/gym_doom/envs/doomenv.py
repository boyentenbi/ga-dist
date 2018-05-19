#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplifie Banana selling environment.
Each episode is selling a single banana.
"""

# core modules
import random
import math

# 3rd party modules
import gym
import numpy as np
from gym import spaces
from gym.utils import colorize, seeding

from vizdoom import DoomGame, ScreenResolution, ScreenFormat, Button, GameVariable, Mode


class DoomEnv(gym.Env):
    """
    Define a simple Banana environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        print("DoomEnv - Version {}".format(self.__version__))
        
        game = DoomGame()
        
        #The Below code is related to setting up the Doom environment
        game.set_doom_scenario_path("basic.wad") #This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        a_size = 3
        self.actions = np.identity(a_size,dtype=bool).tolist()
        #End Doom set-up
        self.env = game
        


        # Define what the agent can do
        self.action_space = spaces.Discrete(3)

        # Observation is the remaining time               
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 1))
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, a):
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
        r = self.env.make_action(self.actions[a]) / 100.0
        d = self.env.is_episode_finished()     
        ammo = self.env.get_game_variable(GameVariable.AMMO2)
        x_pos = self.env.get_game_variable(GameVariable.POSITION_X)
        y_pos = self.env.get_game_variable(GameVariable.POSITION_Y)
        info = {'ammo':ammo, 'x_pos':x_pos, 'y_pos':y_pos}
        
         
        if d:
            #print("Episode is done")
            return np.zeros(self.observation_space.shape), r, d, info

        else:
            s = self._get_screen()
            return s, r, d, info

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.env.new_episode()
        
        return self._get_screen()

    def _render(self, mode='human', close=False):
        
        return self._get_screen()
   
    def _get_screen(self):
        screen = self.env.get_state().screen_buffer
        if len(screen.shape)==2:
            return np.expand_dims(screen, -1)
        elif len(screen.shape)==3:
            return screen
        else:
            raise Exception("screen tensor had rank not equal to 2 or 3")
        
