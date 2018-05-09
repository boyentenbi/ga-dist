import logging
import os
from clients import MasterClient, RelayClient, WorkerClient
from clients import EXP_KEY, TASK_DATA_KEY, TASK_ID_KEY, TASK_CHANNEL, RESULTS_KEY
from multiprocessing import Process
import json
from queue import PriorityQueue
import time
import numpy as np
import csv
import gym
from baselines import deepq
from baselines.common.atari_wrappers import  make_atari, wrap_deepmind

gym.undo_logger_setup()
import policies, tf_util
from collections import namedtuple, OrderedDict
import tf_util as U

import subprocess

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

# Task = namedtuple('Task', [])

Config = namedtuple('Config', [
    "noise_stdev",
    "n_noise",
    "n_tsteps",
    "episodes_per_batch",
    "timesteps_per_batch",
    "init_tstep_limit",
    "min_gen_time",
    "n_parents",
    "n_elite_candidates",
    "n_evals",
    'global_seed',

    'l2coeff',
    'calc_obstat_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode', "adaptive_tstep_lim", 'tstep_lim_incr_ratio',
    'tstep_maxing_thresh',
])

Result = namedtuple('Result', [
    'worker_id',
    'noise_list',
    'ret',
    'n_steps',
    'n_seconds',
    'finish_time',
    'is_eval',
    'gen_num',
])




Task = namedtuple('Task', [
    'parent_noise_lists',
    'timestep_limit',
    'gen_num'
])

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
    def __init__(self, n_nodes, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw, log_dir):

        # Initialize networking
        self.n_nodes = n_nodes
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
        assert self.config.n_elite_candidates <= self.config.n_parents
        self.global_seed = self.config.global_seed

        #self.min_steps = self.config.min_steps
        #self.min_eps = self.config.min_eps
        self.noise = SharedNoiseTable(self.global_seed, self.config.n_noise)
        # Now a genome can be specified with a list of indices
        # Sampling seeds and then using the first sample from resulting normal
        # wouldn't necessarily give normally distributed samples!

        # Start worker processes
        # self.wps = []
        for process_num in range(self.n_workers):
            worker_id = node_id * self.n_workers + process_num
            wp = Process(target= self.worker_process, args=(worker_id,))
            # self.wps.append(wp)
            wp.start()

    def get_n_workers(self):
        raise NotImplementedError
    # def get_master_redis_cfg(self, password):
    #     raise NotImplementedError

    def worker_process(self, worker_id):
        wc = WorkerClient(relay_redis_cfg=self.relay_redis_cfg)

        # Set up the experiment
        if self.exp['policy']['type'] == 'AtariPolicy':
            env = wrap_deepmind(make_atari(self.exp["env_id"]), episode_life=False, clip_rewards=False, frame_stack=True, scale=True)

            # env = wrap_deepmind(make_atari(self.exp['env_id']), episode_life=True, clip_rewards=True, frame_stack=True, scale=True,)
        elif self.exp['policy']['type'] == 'MujocoPolicy':
            env = gym.make(self.exp['env_id'])
        else:
            raise NotImplementedError("This type of policy is not supported yet")
        sess = make_session(single_threaded=True)
        policy_class = getattr(policies, self.exp['policy']['type'])
        policy = policy_class(env.observation_space,
                              env.action_space,
                              **self.exp['policy']['args'])

        tf_util.initialize()
        rs = np.random.RandomState()

        # Allocate candidates to this worker
        my_candidates = []
        for i in range(self.config.n_elite_candidates):
            for j in range(self.config.n_evals):
                if (self.config.n_evals * i + j) % (self.n_nodes * self.n_workers) == worker_id:
                    my_candidates.append(i)

        eps_done = 0
        cached_gen_num = None

        while True:
            # Grab task data
            task_id, task_data = wc.get_current_task()

            assert isinstance(task_id, int) and isinstance(task_data, Task)
            assert isinstance(task_data.parent_noise_lists, list) and isinstance(task_data.timestep_limit, int)
            assert isinstance(task_data.gen_num, int)
            # assert isinstance(task_data.eval_task, bool)

            # new generation, reset candidates_done
            if task_data.gen_num != cached_gen_num:
                candidates_done = 0
                cached_gen_num = task_data.gen_num

            # Prep for rollouts
            #noise_sublists, returns, lengths = [], [], []

            cycle_tstart = time.time()
            cycle_eps_done = 0

            while cycle_eps_done < 1 or time.time() - cycle_tstart < self.config.min_gen_time \
                    or (candidates_done < len(my_candidates) and len(task_data.parent_noise_lists) != 0):

                if len(task_data.parent_noise_lists) == 0:
                    # First generation

                    new_idx = self.noise.sample_index(rs, policy.num_params)
                    noise_list = [new_idx]
                    is_eval = False
                    # Use the first index to initialize using glorot
                    v = policy.init_from_noise_idxs(
                        self.noise.get(noise_list[0], policy.num_params), glorot_std=1.)
                    parent_idx = None

                else:

                    if candidates_done < len(my_candidates):
                        candidate_num = my_candidates[candidates_done]
                        noise_list = task_data.parent_noise_lists[-1 - candidate_num]
                        candidates_done +=1
                        is_eval = True
                    else:
                        parent_idx = rs.choice(len(task_data.parent_noise_lists))
                        parent_noise_list = task_data.parent_noise_lists[parent_idx]
                        new_idx = self.noise.sample_index(rs, policy.num_params)
                        noise_list = parent_noise_list + [new_idx]
                        is_eval = False
                    # Use the first index to initialize using glorot
                    init_params = policy.init_from_noise_idxs(self.noise.get(noise_list[0], policy.num_params), glorot_std=1.)

                    v = init_params
                    for j in noise_list[1:]:
                        v += self.config.noise_stdev * self.noise.get(j, policy.num_params)

                policy.set_trainable_flat(v)
                rewards, n_steps = policy.rollout(env, timestep_limit=self.config.init_tstep_limit)

                # policy.set_trainable_flat(task_data.params - v)
                # rews_neg, len_neg = rollout_and_update_ob_stat(
                #     policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)
                ret = float(np.sum(rewards))
                finish_time = time.time()
                n_seconds = float(finish_time - cycle_tstart)
                logger.info('Eval result: task_id={} return={:.3f} length={}'.format(
                    task_id, ret, n_steps))
                wc.push_result(task_id,
                               Result(worker_id=worker_id,
                                      noise_list=noise_list,
                                      ret=ret,
                                      n_steps=n_steps,
                                      n_seconds=n_seconds,
                                      finish_time=finish_time,
                                      is_eval=is_eval,
                                      gen_num=task_data.gen_num))
                cycle_eps_done += 1

            eps_done += cycle_eps_done

# The master node also has worker processes
class MasterNode(Node):
    def __init__(self, n_nodes, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw, log_dir):

        # Initialize networking
        super().__init__(n_nodes, node_id, n_workers, exp,
                         master_host, master_port, relay_socket, master_pw, log_dir)
        logger.info("Node {} contains the master client.".format(self.node_id))
        self.master_client = MasterClient(self.master_redis_cfg)
        self.log_quantities = [
            'worker_id',
            'noise_list',
            'ret',
            'n_steps',
            'n_seconds',
            'finish_time',
            'is_eval',
            'worker_gen',
            'worker_task_id'
            'master_task_id']

    def begin_exp(self):

        import tabular_logger as tlogger

        # Logging files! Very important!
        #year, month, day, hour, min, sec = time.localtime()[:6]
        #log_folder = "deepmind-{}.{}.{}:{}:{}.{}-{}-{}".format(self.exp['env_id'], self.n_workers, hour, min, sec, day, month, year)
        csv_log_path = os.path.join(self.log_dir, "results.csv")
        tab_log_path = os.path.join(self.log_dir)
        json_log_path = os.path.join(self.log_dir, "short_log.json")
        logger.info('Tabular logging to {}/log.txt'.format(tab_log_path))
        logger.info('csv logging to {}'.format(csv_log_path))
        tlogger.start(tab_log_path)
        with open(csv_log_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_quantities)
            writer.writeheader()
        with open(json_log_path, 'w') as f:
            json.dump([], f)
        # Prepare for experiment
        self.master_client.declare_experiment(self.exp)
        parent_noise_lists, parent_rets = [], []
        candidate_noise_lists = []
        tstep_lim = self.config.init_tstep_limit
        gen_num, n_exp_eps, n_exp_steps = 0, 0, 0
        elite_noise_lists = []
        task_ids_set = set([])
        sess = make_session(single_threaded=True) # don't comment me out!

        #env = wrap_deepmind(make_atari(self.exp['env_id']), frame_stack=True, scale=True, episode_life=False, clip_rewards=False)
        #policy_class = getattr(policies, self.exp['policy']['type'])
        #policy = policy_class(env.observation_space, env.action_space, **self.exp['policy']['args'])
        tf_util.initialize()

        exp_tstart = time.time()
        # Iterate over generations
        while n_exp_steps < self.config.n_tsteps:

            gen_tstart = time.time()

            task_id = self.master_client.declare_task(
                    Task(parent_noise_lists=parent_noise_lists,
                         timestep_limit=self.config.init_tstep_limit,
                         gen_num=gen_num))
            assert task_id not in task_ids_set
            task_ids_set.add(task_id)
            # Collect the generation (a lot of work)
            results = self.collect_gen(task_id, gen_num)


            # results = [r for r in results if r['worker_gen_id'] in task_ids_set]
            # All other nodes are now wasting compute for master from here!
            mut_results =  [r for r in results if r['worker_task_id']==task_id and not r['is_eval']]
            mut_results.sort(key=lambda r: r['ret'])
            eval_results = [r for r in results if r['worker_task_id'] == task_id and r['is_eval']]

            top_results = mut_results[-self.config.n_parents:]
            parent_noise_lists = [r['noise_list'] for r in top_results]

            n_exp_eps += len(mut_results)
            n_exp_steps += sum([r['n_steps'] for r in mut_results])
            gen_num += 1

            # # Determine if the timestep limit needs to be increased
            # if self.config.adaptive_tstep_lim and \
            #         np.mean(lens==tstep_lim) > self.config.tstep_maxing_thresh:
            #     old_tslimit = tslimit
            #     tslimit = int(self.config.tstep_lim_incr_ratio * tslimit)
            #     logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

            # append previous best to new list: 'elitism'

            if gen_num >= 2:
                cnl_strings = ["-".join([str(idx) for idx in cnl]) for cnl in candidate_noise_lists]
                # assert we have enough evals for each candidate
                eval_results_agg = dict(zip(cnl_strings, [{"count":0,
                                                          "rets":[],
                                                          "lens":[]} for _ in range(self.config.n_elite_candidates)]))

                for er in eval_results:
                    cnl_string = "-".join([str(idx) for idx in er["noise_list"]])

                    eval_results_agg[cnl_string]["count"] += 1
                    eval_rets_dict[cnl_string]["rets"].append(er["ret"])
                    eval_lens_dict[cnl_string]["lens"].append(er["len"])

                elite_noise_list = candidate_noise_lists[0] # initialization (not final!)
                top_mean = None
                for cnl_str, data in eval_results_agg:
                    assert data["count"] == self.config.n_evals
                    data["mean_ret"] = np.mean(data["rets"])
                    if not top_mean or data["mean_ret"] >= top_mean:
                        top_mean = data["mean_ret"]
                        elite_noise_list = [int(x) for x in cnl_str.split("-")]
                assert elite_noise_list in candidate_noise_lists

                elite_noise_lists.append(elite_noise_list)
                parent_noise_lists.append(elite_noise_list)  # this is actually the elite from the prev
                # dummy_elite_result = {"noise_list":elite_nl,
                #                       "ret": top_score}
                # sorted_w_elite = sorted(mut_results[-self.config.n_parents:]+[dummy_elite_result],
                #                         key = lambda x: x['ret'])
            else:
                eval_rets_dict = {}
                eval_lens_dict = {}
                elite_noise_list = None
                top_mean = None
            candidate_noise_lists = parent_noise_lists[-self.config.n_elite_candidates:]

            # Compute the skip fraction
            n_bad_eps = len(results)-len(mut_results)-len(eval_results)
            skip_frac = n_bad_eps/ len(results)
            if skip_frac > 0:
                logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                    n_bad_eps, 100. * skip_frac))

            gen_tend = time.time()

            # Write the short logs
            self.log_json(mut_results,  eval_results,
                          eval_rets_dict, eval_lens_dict,
                          elite_noise_list,
                          gen_tend, gen_tstart,  skip_frac, exp_tstart, n_exp_steps, n_exp_eps,
                          json_log_path)

            # Record literally everything
            # This should only be ~ 5MB for an entire experiment
            self.log_csv(results, csv_log_path, gen_num, exp_tstart)


        logger.info("Finished {} generations in {} timesteps.".format(gen_num, n_exp_steps))

        # logger.info("Running elite eval gen")
        # # declare the elite eval gen
        # eval_gen_id =self.master_client.declare_gen(
        #     Gen(done=True,
        #         noise_lists=elite_noise_lists,
        #         timestep_limit=self.config.init_tstep_limit))
        # n_gen_eps, n_gen_steps, n_bad_eps, n_bad_steps, bad_time, \
        # elite_genomes, elite_rets, elite_lens = self.collect_elite_gen(eval_gen_id, gen_start_queue_size, elite_noise_lists, self.config.n_elite_evals)
        # logger.info("Finished elite eval gen with {} episodes.".format(n_gen_eps))
        #
        # # Log the cross-generation elite results
        # elite_log_path = os.path.join(self.log_dir, "elite_log.csv")
        # elite_eval_fieldnames = [
        #         'EpRetMax', 'EpRetUQ', 'EpRetMed', 'EpRetLQ', 'EpRetMin',
        #         'EpLenMax', 'EpLenUQ', 'EpLenMed', 'EpLenLQ', 'EpLenMin',]
        # with open(elite_log_path, 'w') as f:
        #     writer = csv.DictWriter(f, fieldnames=elite_eval_fieldnames)
        #     writer.writeheader()
        # for eg in elite_genomes:
        #     elite_log_dict = {"EpRetMax": np.nan if not elite_rets[eg] else np.max(elite_rets[eg]),
        #     "EpRetUQ": np.nan if not elite_rets[eg] else np.percentile(elite_rets[eg], 75),
        #     "EpRetMed": np.nan if not elite_rets[eg] else np.median(elite_rets[eg]),
        #     "EpRetLQ": np.nan if not elite_rets[eg] else np.percentile(elite_rets[eg], 25),
        #     "EpRetMin": np.nan if not elite_rets[eg] else np.min(elite_rets[eg]),
        #
        #     "EpLenMax": np.nan if not elite_lens[eg] else np.max(elite_lens[eg]),
        #     "EpLenUQ": np.nan if not elite_lens[eg] else np.percentile(elite_lens[eg], 75),
        #     "EpLenMed": np.nan if not elite_lens[eg] else np.median(elite_lens[eg]),
        #     "EpLenLQ": np.nan if not elite_lens[eg] else np.percentile(elite_lens[eg], 25),
        #     "EpLenMin": np.nan if not elite_lens[eg] else np.min(elite_lens[eg]),}
        #     self.csv_log_append(elite_log_dict, elite_log_path, elite_eval_fieldnames)

        if "SLURM_JOB_ID" in os.environ:
            logger.info("Running on cluster. Declaring experiment end. SLURM_JOB_ID = {}".format(os.environ["SLURM_JOB_ID"]))
            subprocess.call("scancel {}".format(os.environ["SLURM_JOB_ID"]), shell=True)
        else:
            logger.info("Running on login node(s). Experiment finished. ")


    # def collect_elite_gen(self, eval_gen_id, gen_start_queue_size, elite_noise_lists, n_elite_evals):
    #
    #     elite_genomes = ["-".join([str(x)for x in enl]) for enl in elite_noise_lists]
    #
    #     elite_rets = OrderedDict(zip(elite_genomes, [[] for eg in elite_genomes]))
    #     elite_lens = OrderedDict(zip(elite_genomes, [[] for eg in elite_genomes]))
    #
    #     # Prep for new gen results
    #     n_gen_eps, n_gen_steps, \
    #     n_bad_eps, n_bad_steps, bad_time = 0, 0, 0, 0, 0
    #
    #     while min([len(rets) for rets in elite_rets.values()])<n_elite_evals:
    #         # Pop a result, accumulate if current gen, throw if past gen
    #         worker_gen_id, r = self.master_client.pop_result()
    #
    #         if worker_gen_id == eval_gen_id:
    #             n_gen_eps += 1
    #             n_gen_steps += r.len
    #             eg = "-".join([str(x)for x in r.noise_list])
    #             elite_lens[eg].append(r.len)
    #             elite_rets[eg].append(r.ret)
    #         else:
    #             n_bad_eps += 1
    #             n_bad_steps += r.len
    #             bad_time += r.time
    #             assert n_bad_eps < gen_start_queue_size + 10000
    #         logger.debug("n_gen_eps = {}, n_bad_eps = {}".format(n_gen_eps, n_bad_eps))
    #
    #     return n_gen_eps, n_gen_steps, n_bad_eps, n_bad_steps, bad_time,\
    #            elite_genomes, elite_rets, elite_lens

    def collect_gen(self, master_task_id, master_gen_num):
        # Prep for new gen results
        n_exp_eps, n_exp_steps = 0, 0

        # n_eval_eps, n_eval_steps = 0, 0
        # n_bad_eps, n_bad_steps, bad_time = 0, 0, 0, 0, 0, 0, 0
        # mut_rets, mut_lens, mut_times, mut_noise_lists = [], [], [], []
        # eval_rets, eval_lens, eval_times, eval_noise_lists = [], [], [], []
        # worker_eps = {}

        results = []


        # assert task_id not in gen_nums.keys()
        # gen_nums[task_id] = gen_num
        # Count the number on the queue immediately after declaring the generation
        # We shouldn't get more than this number of bad episodes
        gen_start_queue_size = self.master_client.master_redis.llen(RESULTS_KEY)

        while n_exp_eps < self.config.episodes_per_batch or \
                n_exp_steps < self.config.timesteps_per_batch:
            # Continue if results not full
            # Do I want to count evals in number of eps done?
            # Yes since used to choose elite
            worker_task_id, r = self.master_client.pop_result()

            result_dict = {'worker_id': r.worker_id,
                            'noise_list': r.noise_list,
                            'ret': r.ret,
                            'n_steps': r.n_steps,
                            'n_seconds': r.n_seconds,
                            'finish_time': r.finish_time,
                            'is_eval': r.is_eval,
                            'worker_gen': r.gen_num,
                            'worker_task_id':worker_task_id,
                            'master_task_id':master_task_id}

            results.append(result_dict)
            if master_task_id == worker_task_id:
                assert r.gen_num == master_gen_num
                n_exp_eps += 1
                n_exp_steps += r.n_steps

            # when one of the eval requirements is done
            # change task


        return results

            # if r.is_eval:
            #
            #     if worker_gen_id == task_id:
            #         worker_id = r.worker_id
            #         if worker_id in worker_eps:
            #             worker_eps[worker_id] += 1
            #         else:
            #             worker_eps[worker_id] = 1
            #         n_eval_eps += 1
            #         n_eval_steps += r.n_steps
            #         eval_rets.append(r.ret)
            #         eval_lens.append(r.n_steps)
            #         eval_noise_lists.append(r.noise_list)
            #         eval_times.append(r.n_millis)
            #     else:
            #         n_bad_eps += 1
            #         n_bad_steps += r.n_steps
            #         bad_time += r.n_millis
            #
            # else:
            #     if worker_gen_id == task_id:
            #         worker_id = r.worker_id
            #         if worker_id in worker_eps:
            #             worker_eps[worker_id] += 1
            #         else:
            #             worker_eps[worker_id] = 1
            #         n_gen_eps += 1
            #         n_gen_steps += r.n_steps
            #         mut_rets.append(r.ret)
            #         mut_lens.append(r.n_steps)
            #         mut_noise_lists.append(r.noise_list)
            #         mut_times.append(r.n_millis)
            #
            #     else:
            #         n_bad_eps += 1
            #         n_bad_steps += r.n_steps
            #         bad_time += r.n_millis
            # logger.debug("n_gen_eps = {}, n_bad_eps = {}".format(n_gen_eps, n_bad_eps))

        # return n_mut_eps, n_mut_steps, \
        #        n_eval_eps, n_eval_steps, \
        #        n_bad_eps, n_bad_steps, bad_time, \
        #        worker_eps, \
        #        mut_rets, mut_lens, mut_times, mut_noise_lists,\
        #        eval_rets, eval_lens, eval_times, eval_noise_lists

    def tabular_log_append(self, log_dict, tlogger):

        for quantity, value in log_dict.items():
            if not isinstance(value, str):
                tlogger.record_tabular(quantity, value)
        tlogger.dump_tabular()

    def log_csv(self, results, csv_log_path, gen_num, exp_tstart):

        saveable_results = []
        for r in results:
            row_dict = {'worker_id'  : r['worker_id'],
                        'noise_list' : "-".join([str(idx) for idx in r['noise_list']]),
                        'ret'        : r['ret'],
                        'n_steps'    : r['n_steps'],
                        'n_seconds'   : r['n_seconds'],
                        'finish_time': r['finish_time']-exp_tstart,
                        'is_eval'    : r['is_eval'],
                        'worker_gen' : r['worker_gen'],
                        'worker_task_id' : r['worker_task_id'],
                        'master_task_id': r['master_task_id'] }
            saveable_results.append(row_dict)

        with open(csv_log_path, 'a') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=[
                                        'worker_id',
                                        'noise_list',
                                        'ret',
                                        'n_steps',
                                        'n_seconds',
                                        'finish_time',
                                        'is_eval',
                                        'worker_gen',
                                        'worker_task_id',
                                        'master_task_id',
                                    ])
            writer.writerows(saveable_results)


    def log_json(self, mut_results, eval_results,
                 eval_rets_dict, eval_lens_dict,
                          elite_noise_list,
                          gen_tend, gen_tstart,  skip_frac, exp_tstart, n_exp_steps, n_exp_eps,
                          json_log_path):

        # eval_rets = [r['ret'] for r in eval_results]
        # eval_lens = [r['n_steps'] for r in eval_results]
        mut_rets = [r['ret'] for r in mut_results]
        mut_lens = [r['n_steps'] for r in mut_results]
        elite_rets = eval_rets_dict[elite_noise_list] if elite_noise_list else None
        elite_lens = eval_lens_dict[elite_noise_list] if elite_noise_list else None

        # eval_eps = [len(x) for x in eval_lens_dict.values()]
        eval_means = [np.mean(x) for x in elite_rets.values()] if elite_rets else None

        unique_workers = set([r['worker_id'] for r in mut_results + eval_results])
        worker_eps = {}
        for w in unique_workers:
            worker_eps[w] = len([r for r in mut_results + eval_results if r['worker_id'] == w])
        weps = np.asarray([x for x in worker_eps.values()])
        d=\
        {'EliteRetMax': None if not elite_rets else float(np.max(elite_rets)),
         'EliteRetMed': None if not elite_rets else float(np.median(elite_rets)),
         'EliteRetMin': None if not elite_rets else float(np.min(elite_rets)),

         'EliteLenMax': None if not elite_lens else int(np.max(elite_lens)),
         'EliteLenMed': None if not elite_lens else int(np.median(elite_lens)),
         'EliteLenMin': None if not elite_lens else int(np.min(elite_lens)),

         # 'EvalEpsMax': None if not eval_eps else np.max(eval_eps),
         # 'EvalEpsMed': None if not eval_eps else np.median(eval_eps),
         # 'EvalEpsMin': None if not eval_eps else np.min(eval_eps),

         'EvalMeanMax': None if not eval_means else np.max(eval_means),
         'EvalMeanMed': None if not eval_means else np.median(eval_means),
         'EvalMeanMin': None if not eval_means else np.min(eval_means),

         # 'EliteSteps': sum(eval_lens),

         "MutRetMax": None if not mut_rets else float(np.max(mut_rets)),
         "MutRetParentMin": None if not mut_rets else float(mut_rets[-self.config.n_parents]),
         "MutRetMed": None if not mut_rets else float(np.median(mut_rets)),
         "MutRetMin": None if not mut_rets else float(np.min(mut_rets)),

         "MutLenMax": None if not mut_lens else int(np.max(mut_lens)),
         "MutLenMed": None if not mut_lens else int(np.median(mut_lens)),
         "MutLenMin": None if not mut_lens else int(np.min(mut_lens)),

         "MutEps": len(mut_results),
         "TimestepsThisIter": sum(mut_lens),
         "ExpMutEps": n_exp_eps,
         "TimestepsSoFar": n_exp_steps,

         "UniqueWorkers": len(unique_workers),
         "UniqueWorkersFrac": float(len(unique_workers)) / float(np.sum(weps)),
         "WorkerEpsMax": int(np.max(weps)),
         "WorkerEpsMed": int(np.median(weps)),
         "WorkerEpsMin": int(np.min(weps)),

         "ResultsSkippedFrac": float(skip_frac),
         "TimeElapsedThisIter": float(gen_tend - gen_tstart),
         "TimeElapsed": float(gen_tend - exp_tstart),

         "EliteGenome": None if not elite_noise_list else "-".join(map(str, elite_noise_list))
         }

        with open(json_log_path, 'r') as f:
            results = json.load(f)
        with open(json_log_path, 'w') as f:
            results.append(d)
            json.dump(results, f, indent=2)

    def get_n_workers(self):
        if self.n_workers:
            assert os.cpu_count - 2 >= self.n_workers
            return min(os.cpu_count() - 2, self.n_workers)
        else:
            return os.cpu_count() - 2

    # def get_master_redis_cfg(self, password):
    #     return {'unix_socket_path': self.socket_path, 'db': 0, "password": password}


class WorkerNode(Node):
    def __init__(self, n_nodes, node_id, n_workers, exp,
                 master_host, master_port, relay_socket, master_pw, log_dir):
        super().__init__(n_nodes, node_id, n_workers, exp,
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

