import logging
import os
from .clients import MasterClient, RelayClient, WorkerClient, \
from .clients import EXP_KEY, GEN_DATA_KEY, GEN_NUM_KEY, TASK_CHANNEL, RESULTS_KEY
from multiprocessing import Process
import json
log = logging.Logger()
log = logging.getLogger(__name__)

import time

from collections import namedtuple

Task = namedtuple('Task', [])

Config = namedtuple('Config', [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode'
])

Result = namedtuple('Result', [
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
])

def setup(exp, single_threaded):
    import gym
    gym.undo_logger_setup()
    from . import policies, tf_util

    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space, env.action_space, **exp['policy']['args'])
    tf_util.initialize()

    return config, env, sess, policy


def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))



class WorkerProcess:
    """
    Sets up both the worker client and evaluation thread
    Runs both concurrently
    """
    def __init__(self, relay_redis_cfg, exp_config):
        wc = WorkerClient(relay_redis_cfg=)
        config, env, sess, policy = setup(exp_config, single_threaded=True)

        while True:
            gen_num, gen_data = wc.get_current_gen()
            gen_tstart = time.time()
            eval_return, eval_length = policy.rollout(env)
            log.info('Eval result: gen={} return={:.3f} length={}'.format(gen_num, eval_return, eval_length))

            wc.push_result(gen_num, Result(gen_num, gen_data))

class Node:
    def __init__(self, node_id, n_workers, exp_config):

        # Initialize networking
        self.node_id = node_id
        assert n_workers <= os.cpu_count()
        self.n_workers = n_workers if n_workers else self.get_n_workers()

        master_redis_cfg = {'host': master_host, 'port': master_port}
        relay_redis_cfg = {'unix_socket_path': relay_socket_path}

        # Relay client process
        rcp = Process(target = lambda : RelayClient(master_redis_cfg, relay_redis_cfg).run())
        rcp.start()

        # Set up the experiment
        self.exp_config = exp_config
        self.global_seed = self.exp_config.global_seed


        # Start worker processes
        for i in range(self.n_workers):
            wp = Process(target= lambda : WorkerProcess())
    def get_n_workers(self):
        raise NotImplementedError


# The master node also works
class MasterNode(Node):
    def __init__(self, n_workers):

        # Initialize networking
        super().__init__(self, n_workers)
        log.info("Node {} contains the master client.".format(self.node_id))
        self.master_client = MasterClient()

        # Start the workers
        seed_lists = [[i] for i in range(len(self.node_list))]
        for i in range(len(self.node_list)):
            self.master_client.redis.set('seed-lists-{}'.format(i), seed_lists)

    def get_n_workers(self):
        if self.n_workers:
            return self.n_workers
        else:
            return os.cpu_count() - 2

class WorkerNode(Node):
    def __init__(self, n_workers):
        super().__init__(self, n_workers)

        log.info("Node {} is a worker node".format(self.node_id))
        assert n_workers <= os.cpu_count()
        self.n_workers = n_workers if n_workers else os.cpu_count() -1

    def get_n_workers(self):
        if self.n_workers:
            return self.n_workers
        else:
            return os.cpu_count() - 1

