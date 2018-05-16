from gym.envs.registration import register

register(
    id='MetaBandit-v0',
    entry_point='gym_meta_bandit.envs:MetaBanditEnv',
)
