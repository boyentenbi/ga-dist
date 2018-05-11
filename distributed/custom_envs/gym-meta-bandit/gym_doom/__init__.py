from gym.envs.registration import register

register(
    id='metabanditenv-v0',
    entry_point='meta_bandit_env.envs:MetaBanditEnv',
)
