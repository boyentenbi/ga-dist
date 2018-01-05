import logging
import os
from clients import MasterClient, RelayClient, WorkerClient
from clients import EXP_KEY, GEN_DATA_KEY, GEN_NUM_KEY, TASK_CHANNEL, RESULTS_KEY, NOISES_KEY
from multiprocessing import Process
import json
log = logging.getLogger(__name__)

import time
import numpy as np

import gym

gym.undo_logger_setup()
import policies, tf_util
from collections import namedtuple

Task = namedtuple('Task', [])

Config = namedtuple('Config', [
    'global_seed', 'n_gens','l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode'
])

Result = namedtuple('Result', [
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
])

def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))

class SharedNoiseTable(object):
    def __init__(self, seed):
        import ctypes, multiprocessing

        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        log.info('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        log.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


class Node:
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, socket_path):

        # Initialize networking
        self.master_port = master_port
        self.node_id = node_id
        if n_workers:
            assert n_workers < os.cpu_count()
        self.n_workers = None
        self.n_workers = n_workers if n_workers else self.get_n_workers()
        self.socket_path = socket_path
        self.relay_redis_cfg = {'unix_socket_path': socket_path, 'db': 1}
        self.master_redis_cfg = self.get_master_redis_cfg()
        self.master_host = master_host
        self.exp = exp

        # Relay client process
        rcp = Process(target = RelayClient(self.master_redis_cfg,
                                           self.relay_redis_cfg).run)
        rcp.start()


        # Initialize the experiments
        self.config = Config(**self.exp['config'])
        self.global_seed = self.config.global_seed
        #self.min_steps = self.config.min_steps
        #self.min_eps = self.config.min_eps
        self.noise = SharedNoiseTable(self.global_seed)
        # Now a genome can be specified with a list of indices
        # Sampling seeds and then using the first sample from resulting normal
        # wouldn't necessarily give normally distributed samples!

        # Start worker processes
        for i in range(self.n_workers):
           wp = Process(target= lambda : self.worker_process)
           wp.start()

    def get_n_workers(self):
        raise NotImplementedError
    def get_master_redis_cfg(self):
        raise NotImplementedError

    def worker_process(self):
        wc = WorkerClient(relay_redis_cfg=self.relay_redis_cfg)
        # Set up the experiment
        env = gym.make(self.exp['env_id'])
        sess = make_session(single_threaded=True)
        policy_class = getattr(policies, self.exp['policy']['type'])
        policy = policy_class(env.observation_space,
                              env.action_space,
                              **self.exp['policy']['args'])
        tf_util.initialize()

        while True:
            gen_num, gen_data = wc.get_current_gen()
            gen_tstart = time.time()
            eval_return, eval_length = policy.rollout(env)
            log.info('Eval result: gen={} return={:.3f} length={}'.format(gen_num, eval_return, eval_length))

            wc.push_result(gen_num, Result(gen_num, gen_data))



# The master node also has worker processes
class MasterNode(Node):
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, master_socket_path):

        # Initialize networking
        super().__init__(node_id, n_workers, exp,
                         master_host, master_port, master_socket_path)
        log.info("Node {} contains the master client.".format(self.node_id))
        self.master_client = MasterClient(self.master_redis_cfg)
        #self.cluster_n_workers = self.exp.n_nodes*(self.n_workers+1)-1

        # TODO think about separate populations to increase CPU utilisation
        #for i in range(len(self.node_list)):
        #    self.master_client.redis.set('noise-lists-{}'.format(i), noise_lists)

    def begin_exp(self):

        self.noise_lists = []
        self.master_client.declare_experiment(self.exp)

        # Iterate over generations
        for gen_num in range(self.config.n_gens):

            workers_done = 0
            results, returns, lens, worker_ids, noise_idxs = [], [], [], [], []
            while workers_done < n_workers:
                worker_gen_num, result = self.master_client.pop_result()
                assert worker_gen_num == gen_num
                workers_done += 1


        # Separate loop for when we change to node-failure-tolerant mode
        for r in results:
            noise_idxs.append(r.noise_idx)
            returns.append(r.ret)
            lens.append(r.len)
            noise_lists[r.worker_id].append(r.noise_idx)

        # Order the returns and use it to choose parents
        # Randomly sample parents n_pop times to form the new population
        # Each new individual is encoded in its noise indices
        order = np.argsort(returns)
        for i in range(self.exp_config.n_pop):
            parent_idx = np.random.choice(order[-self.exp_config.n_parents])
            child_noise_list = noise_lists[parent_idx]

            self.master_client.rpush(NOISES_KEY, child_noise_list) # TODO lpush or rpush? TODO is this the best way to transfer the noise lists?

    def get_n_workers(self):
        if self.n_workers:
            return self.n_workers
        else:
            return os.cpu_count() - 2

    def get_master_redis_cfg(self):
        return {'unix_socket_path': self.socket_path, 'db': 0}


class WorkerNode(Node):
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, socket_path):
        super().__init__(node_id, n_workers, exp,
                         master_host, master_port, socket_path)

        log.info("Node {} is a worker node".format(self.node_id))
        assert n_workers <= os.cpu_count()
        self.n_workers = n_workers if n_workers else os.cpu_count() -1

    def get_n_workers(self):
        if self.n_workers:
            return self.n_workers
        else:
            return os.cpu_count() - 1

    def get_master_redis_cfg(self):
        return {'host': self.master_host, 'port': self.master_port, 'db':0}

