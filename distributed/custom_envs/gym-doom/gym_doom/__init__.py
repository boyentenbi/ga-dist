from gym.envs.registration import register

register(
    id='doomenv-v0',
    entry_point='gym_doom.envs:DoomEnv',
)
