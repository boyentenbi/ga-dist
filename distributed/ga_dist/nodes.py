import logging
import os
from clients import MasterClient, RelayClient, WorkerClient
from clients import EXP_KEY, GEN_DATA_KEY, GEN_ID_KEY, TASK_CHANNEL, RESULTS_KEY
from multiprocessing import Process
import json
from queue import PriorityQueue
import time
import numpy as np
import csv
import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
gym.undo_logger_setup()
import policies, tf_util
from collections import namedtuple
import tf_util as U

import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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




Gen = namedtuple('Gen', ['done', 'noise_lists', 'timestep_limit'])

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
                 master_host, master_port, relay_socket, master_pw, log_dir):

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
        self.log_dir = log_dir

        # Relay client process
        self.rcp = Process(target = RelayClient(self.master_redis_cfg,
                                           self.relay_redis_cfg).run)
        self.rcp.start()

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
        self.wps = []
        for i in range(self.n_workers):
            wp = Process(target= self.worker_process)
            self.wps.append(wp)
            wp.start()



    def get_n_workers(self):
        raise NotImplementedError
    # def get_master_redis_cfg(self, password):
    #     raise NotImplementedError

    def worker_process(self):
        wc = WorkerClient(relay_redis_cfg=self.relay_redis_cfg)
        # Set up the experiment

        env = wrap_deepmind(make_atari(self.exp['env_id']), frame_stack=True, scale=True, episode_life=False, clip_rewards=False)
        sess = make_session(single_threaded=True)
        policy_class = getattr(policies, self.exp['policy']['type'])
        policy = policy_class(env.observation_space,
                              env.action_space,
                              **self.exp['policy']['args'])
        tf_util.initialize()
        rs = np.random.RandomState()
        worker_id = rs.randint(2 ** 31)

        eps_done = 0

        # Keep iterating until gen gets flagged
        while True:
            # Grab generation data
            gen_id, gen_data = wc.get_current_gen()

            if gen_data.done:
                if eps_done == 0 :
                    raise ValueError("!!!Closed worker after 0 completed episodes!!!")
                logger.info("Worker {} finished.".format(worker_id))
                break

            assert isinstance(gen_id, int) and isinstance(gen_data, Gen)
            assert isinstance(gen_data.noise_lists, list) and isinstance(gen_data.timestep_limit, int)

            # Prep for rollouts
            #noise_sublists, returns, lengths = [], [], []
            cycle_tstart = time.time()
            cycle_eps_done = 0

            while cycle_eps_done <1 or time.time() - cycle_tstart < self.config.min_gen_time:

                if len(gen_data.noise_lists) == 0:

                    # Create a list with a single randomly draw index
                    new_idx = self.noise.sample_index(rs, policy.num_params)
                    noise_list = [new_idx]
                    # Use the first index to initialize using glorot
                    v = policy.init_from_noise_idxs(
                        self.noise.get(noise_list[0], policy.num_params))

                    parent_idx = None

                else:
                    parent_idx = rs.choice(len(gen_data.noise_lists))
                    # Sample a noise list from the broadcast ones
                    old_noise_list = gen_data.noise_lists[parent_idx]

                    # Use the remaining indices and one new index to mutate
                    new_idx = self.noise.sample_index(rs, policy.num_params)
                    noise_list = old_noise_list + [new_idx]

                    # Use the first index to initialize using glorot
                    init_params = policy.init_from_noise_idxs(self.noise.get(noise_list[0], policy.num_params))

                    v = init_params
                    for j in noise_list[1:]:
                        v += self.config.noise_stdev * self.noise.get(j, policy.num_params)

                policy.set_trainable_flat(v)
                rewards, length = policy.rollout(env, timestep_limit=self.config.init_tstep_limit)

                # policy.set_trainable_flat(task_data.params - v)
                # rews_neg, len_neg = rollout_and_update_ob_stat(
                #     policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)
                eval_ret = np.sum(rewards)
                duration = time.time() - cycle_tstart
                logger.info('Eval result: gen_id={} return={:.3f} length={} cycle_time = {}'.format(
                    gen_id, eval_ret, length, duration))
                wc.push_result(gen_id, Result(worker_id, noise_list, eval_ret, length, duration))
                cycle_eps_done += 1

                # # This helps to debug
                # # can be left in anyway
                # if cycle_eps_done >= self.config.episodes_per_batch:
                #     logger.info(
                #         "Worker {} finished more episodes than required in total for this batch. Stopping.".format(
                #             worker_id))
                #     break

            eps_done += cycle_eps_done

# The master node also has worker processes
class MasterNode(Node):
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw, log_dir):

        # Initialize networking
        super().__init__(node_id, n_workers, exp,
                         master_host, master_port, relay_socket, master_pw, log_dir)
        logger.info("Node {} contains the master client.".format(self.node_id))
        self.master_client = MasterClient(self.master_redis_cfg)
        self.log_quantities = ['EpRetMax', 'EpRetParentMin', 'EpRetUQ', 'EpRetMed', 'EpRetLQ', 'EpRetMin',
                               'EpLenMax', 'EpLenUQ', 'EpLenMed', 'EpLenLQ', 'EpLenMin',
                               "EpisodesSoFar", "TimestepsThisIter", "TimestepsSoFar",
                               "UniqueWorkers", "UniqueWorkersFrac", "WorkerEpsMax", "WorkerEpsUQ", "WorkerEpsMed",
                               "WorkerEpsLQ", "WorkerEpsMin",
                               "ResultsSkippedFrac",
                               "TimeElapsedThisIter", "TimeElapsed",
                               "TopGenome"]

    def begin_exp(self):

        import tabular_logger as tlogger

        # Logging files! Very important!
        year, month, day, hour, min, sec = time.localtime()[:6]
        #log_folder = "deepmind-{}.{}.{}:{}:{}.{}-{}-{}".format(self.exp['env_id'], self.n_workers, hour, min, sec, day, month, year)
        csv_log_path = os.path.join(self.log_dir, "log.csv")
        tab_log_path = os.path.join(self.log_dir)
        logger.info('Tabular logging to {}/log.txt'.format(tab_log_path))
        logger.info('csv logging to {}'.format(csv_log_path))
        tlogger.start(tab_log_path)
        with open(csv_log_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_quantities)
            writer.writeheader()

        # Prepare for experiment
        exp_tstart = time.time()
        self.master_client.declare_experiment(self.exp)
        noise_lists = []
        tstep_lim = self.config.init_tstep_limit
        n_exp_eps, n_exp_steps = 0, 0
        sess = make_session(single_threaded=True)
        #env = wrap_deepmind(make_atari(self.exp['env_id']), frame_stack=True, scale=True, episode_life=False, clip_rewards=False)
        #policy_class = getattr(policies, self.exp['policy']['type'])
        #policy = policy_class(env.observation_space, env.action_space, **self.exp['policy']['args'])
        tf_util.initialize()

        # Iterate over generations
        for gen_num in range(self.config.n_gens):

            gen_id = self.master_client.declare_gen(
                Gen(done=False,
                    noise_lists=noise_lists,
                    timestep_limit=self.config.init_tstep_limit))

            # Count the number on the queue immediately after declaring the generation
            gen_start_queue_size = self.master_client.master_redis.llen(RESULTS_KEY)
            # We shouldn't get more than this number of bad episodes
            gen_tstart = time.time()

            # Prep for new gen results
            n_gen_eps, n_gen_steps, \
            n_bad_eps, n_bad_steps, bad_time  = 0, 0, 0, 0, 0
            results, returns, lens, mut_noise_lists = [], [], [], []
            worker_eps = {}
            # Keep collecting results until we reach BOTH thresholds
            while n_gen_eps < self.config.episodes_per_batch or \
                    n_gen_steps < self.config.timesteps_per_batch:
                # Pop a result, accumulate if current gen, throw if past gen
                worker_gen_id, r = self.master_client.pop_result()

                if worker_gen_id == gen_id:
                    worker_id = r.worker_id
                    if worker_id in worker_eps:
                        worker_eps[worker_id] += 1
                    else:
                        worker_eps[worker_id] = 1
                    n_gen_eps += 1

                    n_gen_steps += r.len
                    mut_noise_lists.append(r.noise_list)
                    returns.append(r.ret)
                    lens.append(r.len)
                    n_exp_eps +=1
                    n_exp_steps += r.len
                else:
                    n_bad_eps += 1
                    n_bad_steps += r.len
                    bad_time += r.time
                    assert n_bad_eps < gen_start_queue_size + 10000
                logger.info("n_gen_eps = {}, n_bad_eps = {}".format(n_gen_eps, n_bad_eps))
            # All other nodes are now wasting compute for master from here!

            # Determine if the timestep limit needs to be increased
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
            noise_lists = [mut_noise_lists[parent_idx] for parent_idx in parent_idxs]

            # Compute the skip fraction
            skip_frac = n_bad_eps / n_gen_eps
            if skip_frac > 0:
                logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                    n_bad_eps, 100. * skip_frac))

            # stop the clock
            gen_tend = time.time()

            # Write the logs
            log_dict = self.get_log_dict(returns, parent_idxs, lens, n_exp_eps, n_gen_steps, n_exp_steps,
                     worker_eps, gen_tend, exp_tstart, gen_tstart, noise_lists, skip_frac)
            self.tabular_log_append(log_dict, tlogger)
            self.csv_log_append(log_dict, csv_log_path)

        if "SLURM_JOB_ID" in os.environ:
            logger.info("Running on cluster. All generations finished. Declaring experiment end. SLURM_JOB_ID = {}".format(os.environ["SLURM_JOB_ID"]))
            subprocess.call("scancel {}".format(os.environ["SLURM_JOB_ID"]), shell=True)
        else:
            logger.info("Running on login nodes. All generations finished. Declaring experiment end. ")
        # # declare a 'done' gen
        # # TODO replace this with a proper flag
        # self.master_client.declare_gen(
        #     Gen(done=True,
        #         noise_lists=None,
        #         timestep_limit=None))

        # Wait for everything else to close
        # for wp in self.wps:
        #     wp.join()
        # self.rcp.join()
        # subprocess.call("tmux kill-session redis-master", shell=True)
        #
        # logger.info("Node {} (master) has joined all local processes. You may still have to wait for other nodes to close.".format(self.node_id))

    def tabular_log_append(self, log_dict, tlogger):

        for quantity, value in log_dict.items():
            if not isinstance(value, str):
                tlogger.record_tabular(quantity, value)
        tlogger.dump_tabular()

    def csv_log_append(self, log_dict, csv_log_path):
        with open(csv_log_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_quantities)
            writer.writerow(log_dict)

    def get_log_dict(self, returns, parent_idxs, lens, n_exp_eps, n_gen_steps, n_exp_steps,
                     worker_eps, gen_tend, exp_tstart, gen_tstart, noise_lists, skip_frac):

        num_unique_workers = len(worker_eps.keys())
        weps = np.asarray([x for x in worker_eps.values()])
        return \
        {"EpRetMax": np.nan if not returns else np.max(returns),
         "EpRetParentMin": np.nan if not returns else returns[parent_idxs[0]],
         "EpRetUQ": np.nan if not returns else np.percentile(returns, 75),
         "EpRetMed": np.nan if not returns else np.median(returns),
         "EpRetLQ": np.nan if not returns else np.percentile(returns, 25),
         "EpRetMin": np.nan if not returns else np.min(returns),

         "EpLenMax": np.nan if not lens else np.max(lens),
         "EpLenUQ": np.nan if not lens else np.percentile(lens, 75),
         "EpLenMed": np.nan if not lens else np.median(lens),
         "EpLenLQ": np.nan if not lens else np.percentile(lens, 25),
         "EpLenMin": np.nan if not lens else np.min(lens),

         "EpisodesSoFar": n_exp_eps,
         "TimestepsThisIter": n_gen_steps,
         "TimestepsSoFar": n_exp_steps,

         "UniqueWorkers": num_unique_workers,
         "UniqueWorkersFrac": num_unique_workers / np.sum(weps),
         "WorkerEpsMax": np.max(weps),
         "WorkerEpsUQ": np.percentile(weps, 75),
         "WorkerEpsMed": np.median(weps),
         "WorkerEpsLQ": np.percentile(weps, 25),
         "WorkerEpsMin": np.min(weps),

         "ResultsSkippedFrac": skip_frac,
         "TimeElapsedThisIter": gen_tend - gen_tstart,
         "TimeElapsed": gen_tend - exp_tstart,

         "TopGenome": "-".join(map(str, noise_lists[-1]))}

    def get_n_workers(self):
        if self.n_workers:
            assert os.cpu_count - 2 >= self.n_workers
            return min(os.cpu_count() - 2, self.n_workers)
        else:
            return os.cpu_count() - 2

    # def get_master_redis_cfg(self, password):
    #     return {'unix_socket_path': self.socket_path, 'db': 0, "password": password}


class WorkerNode(Node):
    def __init__(self, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw, log_dir):
        super().__init__(node_id, n_workers, exp,
                         master_host, master_port, relay_socket, master_pw, log_dir)

        logger.info("Node {} is a worker node".format(self.node_id))
        assert n_workers <= os.cpu_count()
        self.n_workers = n_workers if n_workers else os.cpu_count() -1

        # for wp in self.wps:
        #     wp.join()
        # self.rcp.join()
        #
        # subprocess.call("tmux kill-session redis-relay", shell=True)
        #
        # logger.info("Node {} has joined all processes. Experiment finished.".format(self.node_id))

    def get_n_workers(self):
        if self.n_workers:
            assert os.cpu_count - 1 >= self.n_workers
            return min(self.n_workers, os.cpu_count()-1)
        else:
            return os.cpu_count() - 1

    # def get_master_redis_cfg(self, password):
    #     return {'host': self.master_host, 'port': self.master_port, 'db':0, 'password':password}

