from gym.envs.registration import register

register(
    id='MetaBanditEnv-v0',
    entry_point='gym_meta_bandit.envs:MetaBanditEnv',
)
