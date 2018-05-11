import json
#import redis
import logging
import os
import hostlist
from nodes import MasterNode, WorkerNode
from clients import MasterClient, RelayClient, WorkerClient
import argparse
import os
import subprocess
import re
import gym

import socket
hostname = socket.gethostname()

def _parse_node_list(node_list):
    return hostlist.expand_hostlist(node_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start a master or worker node for the GA experiment')
    parser.add_argument('--env_id', action="store",
                        help='the id of the environment')
    parser.add_argument('--super_exp_id', action="store",
                        help='id for the super experiment')
    parser.add_argument('--global_seed', action="store",
                        help='global seed for the experiment')
    args = parser.parse_args()

    node_name = os.environ['SLURMD_NODENAME']
    node_list = _parse_node_list(os.environ['SLURM_JOB_NODELIST'])
    node_id = node_list.index(node_name)
    #assert len(node_list) == 2
    master_node_name = node_list[0]

    # Load the experiment from file
    if os.environ["SLURMD_NODENAME"]=="login-e-13":
        with open("configurations/atari_login_node.json", 'r') as f:
            exp = json.loads(f.read())
    else:
        with open("configurations/atari.json", 'r') as f:
            exp = json.loads(f.read())

    exp["config"]["global_seed"] = int(args.global_seed)
    exp["env_id"] = args.env_id
    exp["super_exp_id"] = args.super_exp_id

    log_dir = os.path.join("logs", exp["super_exp_id"], exp["env_id"], str(exp["config"]["global_seed"]))

    log_level = logging.DEBUG if os.environ["SLURMD_NODENAME"]=="login-e-13" else logging.INFO
    logging.basicConfig(filename=os.path.join(log_dir, "node_{}.txt".format(node_id)), level=log_level, filemode="w")
    logger = logging.getLogger(__name__)
    logger.info("Starting GA. super_exp_id: {}, env_id: {}, global_seed:{}, node name: {}, assigned id: {}, node list: {}".format(
        exp["super_exp_id"], exp["env_id"], str(exp["config"]["global_seed"]), node_name, node_id, node_list
    ))

    # start the master redis server
    cp = subprocess.run("pkill redis", shell=True)
    if node_id==0 or args.super_exp_id=="login_node_test":
        cp = subprocess.run("tmux kill-session -t redis-master", shell=True)
        cp = subprocess.run("tmux new -s redis-master -d", shell=True)
        cp = subprocess.run("tmux send-keys -t redis-master \"redis-server redis_config/redis_master.conf\" C-m", shell=True)
    # start the relay redis server
    cp = subprocess.run("tmux kill-session -t redis-relay", shell=True)
    cp = subprocess.run("tmux new -s redis-relay -d", shell=True)
    cp = subprocess.run("tmux send-keys -t redis-relay \"redis-server redis_config/redis_local_mirror.conf\" C-m", shell=True)

    if node_id==0:
        # This node contains the master
        master_node = MasterNode(
                len(node_list),
                node_id,
                26 if args.super_exp_id == "login_node_test" else 31, # NOT 32!
                exp,
                master_host=node_name,
                master_port=6379,
                relay_socket='/tmp/es_redis_relay.sock',
                master_pw="deepbrickwindowattack",
                log_dir=log_dir)

        master_node.begin_exp()

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

        logger.info("Node {} joining server.".format(node_id))

    print("Node {} done!".format(node_id))


