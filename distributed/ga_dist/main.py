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

    # parser = argparse.ArgumentParser(description='Start a master or worker node for the GA experiment')
    # parser.add_argument('--env_id', action="store",
    #                     help='the id of the environment')
    # parser.add_argument('--super_exp_id', action="store",
    #                     help='id for the super experiment')
    # args = parser.parse_args()

    node_name = os.environ['SLURMD_NODENAME']
    node_list = _parse_node_list(os.environ['SLURM_JOB_NODELIST'])
    node_id = node_list.index(node_name)
    #assert len(node_list) == 2
    master_node_name = node_list[0]

    # Load the experiment from file
    with open("configurations/atari123.json", 'r') as f:
        exp_config = json.loads(f.read())
    exp_config["env_id"] = os.environ["env_id"]
    if exp_config["env_id"] == "-":
        raise ValueError("This environment is not available from OpenAI gym!")

    log_dir = os.path.join("logs", os.environ["super_exp_id"], exp_config["env_id"])

    logging.basicConfig(filename=os.path.join(log_dir, "node_{}.txt".format(node_id)), level=logging.DEBUG, filemode="w")
    logger = logging.getLogger(__name__)
    logger.info("Starting GA. super_exp_id: {}, env_id: {}, node name: {}, assigned id: {}, node list: {}".format(
        os.environ["super_exp_id"], os.environ["env_id"], node_name, node_id, node_list
    ))

    # start the master redis server
    if node_id==0:
        cp = subprocess.run("tmux new -s redis-master -d", shell=True)
        cp = subprocess.run("tmux send-keys -t redis-master \"redis-server redis_config/redis_master.conf\" C-m", shell=True)
    # start the relay redis server
    cp = subprocess.run("tmux new -s redis-relay -d", shell=True)
    cp = subprocess.run("tmux send-keys -t redis-relay \"redis-server redis_config/redis_local_mirror.conf\" C-m", shell=True)

    if node_id==0:
        # This node contains the master
        master_node = MasterNode(
            node_id,
            31,
            exp_config,
            master_host=node_name,
            master_port=6379,
            relay_socket='/tmp/es_redis_relay.sock',
            master_pw= "deepbrickwindowattack",
            log_dir=log_dir)
        master_node.begin_exp()

    else:
        # start the workers subscriptions
        node = WorkerNode(node_id,
                          31,
                          exp_config,
                          master_host=master_node_name,
                          master_port=6379,
                          relay_socket='/tmp/es_redis_relay.sock',
                          master_pw= "deepbrickwindowattack",
                          log_dir=log_dir)
        logger.info("Node {} joining server.".format(node_id))

    print("Node {} done!".format(node_id))


