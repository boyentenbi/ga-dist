import logging
import os
from clients import MasterClient, RelayClient, WorkerClient
from clients import EXP_KEY, GEN_DATA_KEY, GEN_NUM_KEY, TASK_CHANNEL, RESULTS_KEY
from multiprocessing import Process
import json
from queue import PriorityQueue
import time
import numpy as np

import gym

gym.undo_logger_setup()
import policies, tf_util
from collections import namedtuple
import tf_util as U
Task = namedtuple('Task', [])

Config = namedtuple('Config', [
    'global_seed', 'n_gens', 'n_nodes', 'init_tstep_limit','min_gen_time',
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode', "adaptive_tstep_lim", 'tstep_lim_incr_ratio',
    'trunc_frac', 'tstep_maxing_thresh', 'n_noise'
])

Result = namedtuple('Result', [
    'worker_id',
    'noise_list',
    'ret',
    'len',
    'time'
])

Gen = namedtuple('Gen', ['noise_lists', 'timestep_limit'])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))

class SharedNoiseTable(object):
    def __init__(self, seed, n_noise):
        import ctypes, multiprocessing

        count = n_noise#25*10**7  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        logger.info('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


class Node:
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw):

        # Initialize networking
        self.master_port = master_port
        self.node_id = node_id
        if n_workers:
            assert n_workers < os.cpu_count()
        self.n_workers = None
        self.n_workers = n_workers if n_workers is not None else self.get_n_workers()
        self.relay_socket = relay_socket
        self.relay_redis_cfg = {'unix_socket_path': relay_socket, 'db': 1, }
        self.master_redis_cfg = {'host': master_host, 'db': 0, 'password': master_pw}
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
        self.noise = SharedNoiseTable(self.global_seed, self.config.n_noise)
        # Now a genome can be specified with a list of indices
        # Sampling seeds and then using the first sample from resulting normal
        # wouldn't necessarily give normally distributed samples!

        # Start worker processes
        for i in range(self.n_workers):
           wp = Process(target= self.worker_process)
           wp.start()

    def get_n_workers(self):
        raise NotImplementedError
    # def get_master_redis_cfg(self, password):
    #     raise NotImplementedError

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
        rs = np.random.RandomState()
        worker_id = rs.randint(2 ** 31)

        # Keep iterating through generations infinitely?
        while True:
            # Grab generation data
            gen_num, gen_data = wc.get_current_gen()
            assert isinstance(gen_num, int) and isinstance(gen_data, Gen)
            ep_tstart = time.time()

            # Prep for rollouts
            #noise_sublists, returns, lengths = [], [], []
            eps_done = 0
            # loop if the number of eps is
            while eps_done <1 or time.time() - ep_tstart < self.config.min_gen_time:

                if len(gen_data.noise_lists) == 0 :

                    # Create a list with a single randomly draw index
                    noise_list = [self.noise.sample_index(rs, policy.num_params)]
                    # Use the first index to initialize using glorot
                    v = policy.glorot_flat_w_idxs(
                        self.noise.get(noise_list[0], policy.num_params), std=1.)
                else:
                    # Sample a noise list from the broadcast ones
                    old_noise_list = gen_data.noise_lists[rs.choice(len(gen_data.noise_lists))]

                    # Use the remaining indices and one new index to mutate
                    new_noise_idx = self.noise.sample_index(rs, policy.num_params)
                    noise_list = old_noise_list + [new_noise_idx]

                    # Use the first index to initialize using glorot
                    init_params = policy.glorot_flat_w_idxs(self.noise.get(noise_list[0], policy.num_params), std=1.)

                    v = init_params
                    for j in noise_list[1:]:
                        v += self.config.noise_stdev * self.noise.get(j, policy.num_params)

                policy.set_trainable_flat(v)
                rewards, length = policy.rollout(env)

                # policy.set_trainable_flat(task_data.params - v)
                # rews_neg, len_neg = rollout_and_update_ob_stat(
                #     policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)
                eval_ret = np.sum(rewards)
                duration = time.time() - ep_tstart
                logger.info('Eval result: gen={} return={:.3f} length={} time = {}'.format(
                    gen_num, eval_ret, length, duration))
                wc.push_result(gen_num, Result(worker_id, noise_list, eval_ret, length, duration))
                eps_done += 1

                # This helps to debug
                # can be left in anyway
                if eps_done >=self.config.episodes_per_batch:
                    logger.info("Worker {} finished more episodes than required in total for this batch. Stopping.".format(worker_id))
                    break

# The master node also has worker processes
class MasterNode(Node):
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw):

        # Initialize networking
        super().__init__(node_id, n_workers, exp,
                         master_host, master_port, relay_socket, master_pw)
        logger.info("Node {} contains the master client.".format(self.node_id))
        self.master_client = MasterClient(self.master_redis_cfg)
        self.cluster_n_workers = self.config.n_nodes*(self.n_workers+1)-1

        # TODO think about separate populations to increase CPU utilisation
        #for i in range(len(self.node_list)):
        #    self.master_client.redis.set('noise-lists-{}'.format(i), noise_lists)

    def begin_exp(self, log_dir):

        exp_tstart = time.time()
        noise_lists = []
        self.master_client.declare_experiment(self.exp)
        import tabular_logger as tlogger
        logger.info('Tabular logging to {}'.format(log_dir))
        tlogger.start(log_dir)

        tstep_lim = self.config.init_tstep_limit
        n_exp_eps, n_exp_steps = 0, 0

        env = gym.make(self.exp['env_id'])
        sess = make_session(single_threaded=True)
        policy_class = getattr(policies, self.exp['policy']['type'])
        policy = policy_class(env.observation_space,
                              env.action_space,
                              **self.exp['policy']['args'])
        tf_util.initialize()
        # Iterate over generations
        for gen_num in range(self.config.n_gens):

            #print("Before declaring gen there are {} items on the queue.".format(self.master_client.master_redis.llen(RESULTS_KEY)))
            # Declare the gen
            self.master_client.declare_gen(
                Gen(noise_lists=noise_lists,
                    timestep_limit=self.config.init_tstep_limit))
            # Count the number on the queue immediately after declaring the generation
            gen_start_queue_size = self.master_client.master_redis.llen(RESULTS_KEY)
            # We shouldn't get more than this number of bad episodes
            gen_tstart = time.time()
            # Prep for new gen results
            n_gen_eps, n_gen_steps, \
            n_bad_eps, n_bad_steps, bad_time  = 0, 0, 0, 0, 0
            results, returns, lens, noise_lists = [], [], [], []
            worker_eps = {}
            # Keep collecting results until we reach BOTH thresholds
            while n_gen_eps < self.config.episodes_per_batch or \
                    n_gen_steps < self.config.timesteps_per_batch:
                # Pop a result, accumulate if current gen, throw if past gen
                worker_gen_num, r = self.master_client.pop_result()

                if worker_gen_num == gen_num:
                    worker_id = r.worker_id
                    if worker_id in worker_eps:
                        worker_eps[worker_id] += 1
                    else:
                        worker_eps[worker_id] = 1
                    n_gen_eps += 1

                    n_gen_steps += r.len
                    noise_lists.append(r.noise_list)
                    returns.append(r.ret)
                    lens.append(r.len)
                    n_exp_eps +=1
                    n_exp_steps += 1
                else:
                    n_bad_eps += 1
                    n_bad_steps += r.len
                    bad_time += r.time
                    assert n_bad_eps < gen_start_queue_size + 10000
                logger.info("n_gen_eps = {}, n_bad_eps = {}".format(n_gen_eps, n_bad_eps))
            # All other nodes are now wasting compute for master from here!

            # Determine if the timestep limit needs to be increased

            # Update number of steps to take
            if self.config.adaptive_tstep_lim and \
                    np.mean(lens==tstep_lim) > self.config.tstep_maxing_thresh:
                old_tslimit = tslimit
                tslimit = int(self.config.tstep_lim_incr_ratio * tslimit)
                logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

            # Order the returns and use it to choose parents
            # Random sampling is done in worker processes
            # set the noise list ready for next iter
            order = sorted(range(n_gen_eps), key = lambda x: returns[x])
            parent_idxs = order[-int(self.config.trunc_frac*self.config.episodes_per_batch):]
            noise_lists = [noise_lists[parent_idx] for parent_idx in parent_idxs]

            # Compute the skip fraction
            skip_frac = n_bad_eps / n_gen_eps
            if skip_frac > 0:
                logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                    n_bad_eps, 100. * skip_frac))

            # stop the clock
            gen_tend = time.time()

            # Reward distribution
            tlogger.record_tabular("EpRetMax", np.nan if not returns else np.max(returns))
            tlogger.record_tabular("EpRetMinParent", np.nan if not returns else returns[parent_idxs[0]])

            tlogger.record_tabular("EpRetUQ", np.nan if not returns else np.percentile(returns, 75))
            tlogger.record_tabular("EpRetMed", np.nan if not returns else np.median(returns))
            tlogger.record_tabular("EpRetLQ", np.nan if not returns else np.percentile(returns, 25))
            tlogger.record_tabular("EpRetMin", np.nan if not returns else np.min(returns))

            # Ep len distribution
            tlogger.record_tabular("EpLenMax", np.nan if not lens else np.max(lens))
            tlogger.record_tabular("EpLenUQ", np.nan if not lens else np.percentile(lens, 75))
            tlogger.record_tabular("EpLenMed", np.nan if not lens else np.median(lens))
            tlogger.record_tabular("EpLenLQ", np.nan if not lens else np.percentile(lens, 25))
            tlogger.record_tabular("EpLenMin", np.nan if not lens else np.min(lens))

            # tlogger.record_tabular("EvalPopRank", np.nan if not returns else (
            #         np.searchsorted(np.sort(returns_n2.ravel()), returns).mean() / returns_n2.size))
            tlogger.record_tabular("EpCount", n_gen_eps)

            # Parent reward distribution


            #tlogger.record_tabular("Norm", float(np.square(policy.get_trainable_flat()).sum()))

            #tlogger.record_tabular("EpisodesThisIter", n_gen_eps)
            tlogger.record_tabular("EpisodesSoFar", n_exp_eps)
            tlogger.record_tabular("TimestepsThisIter", n_gen_steps)
            tlogger.record_tabular("TimestepsSoFar", n_exp_steps)

            num_unique_workers = len(worker_eps.keys())
            weps = np.asarray([x for x in worker_eps.values()])
            tlogger.record_tabular("UniqueWorkers", num_unique_workers)
            tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / np.sum(weps))
            tlogger.record_tabular("WorkerEpsMax", np.max(weps))
            tlogger.record_tabular("WorkerEpsUQ", np.percentile(weps, 75))
            tlogger.record_tabular("WorkerEpsMed", np.median(weps))
            tlogger.record_tabular("WorkerEpsLQ", np.percentile(weps,25))
            tlogger.record_tabular("WorkerEpsMin", np.min(weps))

            tlogger.record_tabular("ResultsSkippedFrac", skip_frac)
            #tlogger.record_tabular("ObCount", ob_count_this_batch)

            tlogger.record_tabular("TimeElapsedThisIter", gen_tend - gen_tstart)
            tlogger.record_tabular("TimeElapsed", gen_tend - exp_tstart)
            tlogger.dump_tabular()

    def get_n_workers(self):
        if self.n_workers:
            return self.n_workers
        else:
            return os.cpu_count() - 2

    # def get_master_redis_cfg(self, password):
    #     return {'unix_socket_path': self.socket_path, 'db': 0, "password": password}


class WorkerNode(Node):
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw):
        super().__init__(node_id, n_workers, exp,
                         master_host, master_port, relay_socket, master_pw)

        logger.info("Node {} is a worker node".format(self.node_id))
        assert n_workers <= os.cpu_count()
        self.n_workers = n_workers if n_workers else os.cpu_count() -1

    def get_n_workers(self):
        if self.n_workers:
            return self.n_workers
        else:
            return os.cpu_count() - 1

    # def get_master_redis_cfg(self, password):
    #     return {'host': self.master_host, 'port': self.master_port, 'db':0, 'password':password}

