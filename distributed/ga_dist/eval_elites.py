import os
import pandas
from nodes import SharedNoiseTable
from baselines import deepq

from baselines.common.atari_wrappers import  make_atari, wrap_deepmind
import gym
import policies, tf_util
import json
import numpy as np
from inspect_functions import make_gif
import logging
logger = logging.getLogger(__name__)

super_exp_id = "2018:03:03-21:42:18"
env_id = "SkiingNoFrameskip-v4"
global_seed = 0
n_runs = 100
n_gifs = 5

def eval_elite(super_exp_id, env_id, global_seed, n_runs, n_gifs):
    # Load the experiment from file
    with open("configurations/atari.json", 'r') as f:
        exp = json.loads(f.read())
    exp_config = exp["config"]
    exp_config["env_id"] = env_id
    exp_config["super_exp_id"] = super_exp_id
    exp_config["global_seed"] = global_seed
    exp_path = os.path.join(exp_config["super_exp_id"], exp_config["env_id"], str(exp_config["global_seed"]))
    # elite_meds = pandas.read_csv(os.path.join('logs/', exp_path, 'log.csv'))["TopGenome"]

    elite_genomes = pandas.read_csv(os.path.join('logs/', exp_path, 'log.csv'))["TopGenome"]
    elite_medians = pandas.read_csv(os.path.join('logs/', exp_path, 'elite_log.csv'))["EpRetMed"]
    best_elite_genome = elite_genomes[elite_medians.idxmax(1)]
    noise_list = [int(x) for x in best_elite_genome.split("-")]

    noise = SharedNoiseTable(exp_config["global_seed"], exp_config["n_noise"])

    env = wrap_deepmind(make_atari(exp_config["env_id"]), episode_life=False, clip_rewards=False, frame_stack=True,
                        scale=True, )
    sess = make_session(single_threaded=True)
    policy = policies.AtariPolicy(env.observation_space,
                          env.action_space,
                          **exp['policy']['args'])

    tf_util.initialize()
    rs = np.random.RandomState()

    # Use the first index to initialize using glorot
    init_params = policy.init_from_noise_idxs(noise.get(noise_list[0], policy.num_params), glorot_std=1.)

    v = init_params
    for j in noise_list[1:]:
        v += exp_config["noise_stdev"] * noise.get(j, policy.num_params)


    policy.set_trainable_flat(v)
    total_reward = 0
    for k in range(n_runs):
        rewards, length, obs = policy.rollout(env, timestep_limit=exp_config["init_tstep_limit"], save_obs=True)
        total_reward += np.sum(rewards)
        frames = np.asarray([ob._out[:,:,-1] for ob in obs])
        if k < n_gifs:
            make_gif(frames, os.path.join('logs/', exp_path, 'ep-{}.gif'.format(np.sum(rewards))),
                     duration = length/100, true_image = False, salience = False, salIMGS = None)
    print("mean reward = {}".format(total_reward/n_runs))

def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))

eval_elite(super_exp_id, env_id, global_seed, n_runs, n_gifs)