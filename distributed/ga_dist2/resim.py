import os
import logging
logger = logging.getLogger(__name__)

import pandas as pd

import json
import logging
import hostlist
from nodes import MasterNode, WorkerNode
import argparse
import subprocess
import sys
import socket
import tf_util
hostname = socket.gethostname()

def _parse_node_list(node_list):
    return hostlist.expand_hostlist(node_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start a master or worker node for the GA experiment')
    parser.add_argument('--mode', action="store",
                        help='last_muts|top_elite')
    parser.add_argument('--worker_min', action="store",
                        help='min number of muts per worker')
    args = parser.parse_args()

    node_name = os.environ['SLURMD_NODENAME']
    node_list = _parse_node_list(os.environ['SLURM_JOB_NODELIST'])
    node_id = node_list.index(node_name)
    master_node_name = node_list[0]

    super_exp_id = "2018:05:12-11:43:12"
    super_exp_path = os.path.join("/home/pc517/ga-dist/distributed/logs", super_exp_id)
    n_gifs = 1
    nodes_started = False

    if args.mode == "last_muts":
        env_ids = ["KangarooNoFrameskip-v4"]
        config_file = 'atari_resim_last_muts.json'

    elif args.mode == "top_elite":
        env_ids = os.listdir(super_exp_path)

        config_file = 'atari_resim_top_elite.json'

    else:
        raise Exception("invalid mode")

    for env_id in env_ids:

        if args.mode == "last_muts":
            seeds = [6]
        elif args.mode == "top_elite":
            seeds = os.listdir(os.path.join(super_exp_path, env_id))
        else:
            raise Exception("invalid mode")
        for seed in seeds:

            log_dir = os.path.join(super_exp_path, env_id, str(seed))
            if args.mode == "last_muts":
                log_dir = os.path.join(log_dir, "resim_last_muts")
            elif args.mode == "top_elite":
                log_dir = os.path.join(log_dir, "resim_top_elite")
                if os.path.isfile(os.path.join(log_dir, "elite_short_log.json")):
                    with open(os.path.join(log_dir, "elite_short_log.json"), 'r') as g:
                        if json.load(g):
                            continue
            else:
                raise Exception("Invalid mode")
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            log_level = logging.INFO # if os.environ["SLURMD_NODENAME"]=="login-e-13" else logging.INFO
            node_log_file = os.path.join(log_dir, "node_{}.txt".format(node_id))
            # node_stderr_file = os.path.join(log_dir, "node_{}_err.txt".format(node_id))
            logging.basicConfig(filename=node_log_file, level=log_level, filemode="w")
            logger = logging.getLogger(__name__)
            open(node_log_file, 'w').close()
            # open(node_stderr_file, 'w').close()

            with open(node_log_file, 'a') as f:
                sys.stderr = f
                try:
                    with open("configurations/{}".format(config_file), 'r') as f:
                        exp = json.loads(f.read())

                    exp["config"]["global_seed"] = int(seed)
                    exp["env_id"] = env_id

                    exp_path = os.path.join("/home/pc517/ga-dist/distributed/logs/2018:05:12-11:43:12", env_id, str(seed))

                    # start the master redis server
                    cp = subprocess.run("pkill redis", shell=True)
                    if node_id == 0:
                        cp = subprocess.run("tmux kill-session -t redis-master", shell=True)
                        cp = subprocess.run("tmux new -s redis-master -d", shell=True)
                        cp = subprocess.run(
                            "tmux send-keys -t redis-master \"redis-server redis_config/redis_master.conf\" C-m",
                            shell=True)
                    # start the relay redis server
                    cp = subprocess.run("tmux kill-session -t redis-relay", shell=True)
                    cp = subprocess.run("tmux new -s redis-relay -d", shell=True)
                    cp = subprocess.run(
                        "tmux send-keys -t redis-relay \"redis-server redis_config/redis_local_mirror.conf\" C-m",
                        shell=True)

                    if node_id==0:
                        # This node contains the master
                        node = MasterNode(
                                len(node_list),
                                node_id,
                                20,
                                exp,
                                master_host=node_name,
                                master_port=6379,
                                relay_socket='/tmp/es_redis_relay.sock',
                                master_pw="deepbrickwindowattack",
                                log_dir=log_dir)
                        node.start_workers(n_gifs=n_gifs, gif_path=log_dir)

                        # Get the short log file
                        with open(os.path.join(exp_path, "short_log.json"), 'r') as g:
                            short_logs = json.load(g)

                        if args.mode == "last_muts":
                            # find the final elite genome
                            penult_elite_str = short_logs[-2]["EliteGenome"]
                            penult_elite_nl = [int(x) for x in penult_elite_str.split("-")]
                            noise_str = penult_elite_str
                        elif args.mode == "top_elite":
                            # find the best elite genome
                            elites_str_mean = [(l["EliteGenome"], l["EvalMeanMax"]) for l in short_logs if
                                               l["EliteGenome"]]
                            best_elite_str, best_elite_mean = sorted(elites_str_mean, key=lambda x: x[1])[-1]

                            noise_str = best_elite_str
                        else:
                            raise Exception("bad argument for mode of elite eval")

                        noise_list = [int(x) for x in noise_str.split("-")]

                        logger.info(
                                "Starting elite eval. Super_exp_id: {}, env_id: {}, global_seed:{}, node name: {}, assigned id: {}, node list: {}".format(
                                        super_exp_id, env_id, str(exp["config"]["global_seed"]), node_name, node_id,
                                        node_list
                                ))
                        if args.mode == "last_muts":
                            # load the results file
                            results_path = os.path.join(exp_path, "results.csv")
                            results = pd.read_csv(results_path)
                            # filter down to final penultimate gen
                            penult_gen_num = results["worker_gen"].max() - 1
                            penult_gen = results[results["worker_gen"] == penult_gen_num]
                            valid_penults = penult_gen[penult_gen["is_valid"]]
                            valid_pen_muts = valid_penults[valid_penults["is_eval"] == False].sort_values(by="ret")
                            valid_pen_evals = valid_penults[valid_penults["is_eval"]].sort_values(by="ret")
                            # parent noise lists in final gen are the top results from this gen
                            cand_sums = valid_pen_evals[["noise_list", "ret"]].groupby("noise_list").sum()
                            elite_str = cand_sums.sort_values(by="ret").iloc[-1].name
                            parent_strs = list(valid_pen_muts.iloc[-20:]["noise_list"]) + [elite_str]
                            parent_noise_lists = [[int(x) for x in s.split("-")] for s in parent_strs]
                            node.resim_muts(parent_noise_lists)
                        else:
                            node.eval_elites([noise_list])

                    else:
                        # start the workers subscriptions
                        node = WorkerNode(
                                len(node_list),
                                node_id,
                                31, # NOT 32!
                                exp,
                                master_host=master_node_name,
                                master_port=6379,
                                relay_socket='/tmp/es_redis_relay.sock',
                                master_pw="deepbrickwindowattack",
                                log_dir=log_dir)
                        node.start_workers(n_gifs=n_gifs, gif_path=log_dir)

                    for wp in node.wps:
                        wp.terminate()
                    node.rcp.terminate()
                    tf_util.reset()

                    logger.info("Node {} done!".format(node_id))

                except Exception as e:
                    logger.error(e, exc_info=True)
                    for wp in node.wps:
                        wp.terminate()
                    node.rcp.terminate()
                    if node_id != 0:
                        logger.error("Exception caused node parent process to exit. Exiting... ")
                    else:
                        logger.error("Fatal exception in master node. Exiting...")
                    sys.exit()




def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))

