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


import socket
hostname = socket.gethostname()

def _parse_node_list(node_list):
    return hostlist.expand_hostlist(node_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start a master or worker node for the GA experiment')
    parser.add_argument('--log', action="store",
                        help='the directory to send all logs')
    args = parser.parse_args()

    node_name = os.environ['SLURMD_NODENAME']
    node_list = _parse_node_list(os.environ['SLURM_JOB_NODELIST'])
    node_id = node_list.index(node_name)
    #assert len(node_list) == 2
    master_node_name = node_list[0]

    logging.basicConfig(filename=os.path.join(args.log, "node_{}.txt".format(node_id)), level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("node name: {}, assigned id: {}, node list: {}".format(
        node_name, node_id, node_list
    ))




    if node_id==0:
        cp = subprocess.run("tmux new -s redis-master -d", shell=True)
        cp = subprocess.run("tmux send-keys -t redis-master \"redis-server redis_config/redis_master.conf\" C-m", shell=True)
        logger.debug(cp.stdout)
        logger.debug(cp.stderr)

    # start the relay redis server
    cp = subprocess.run("tmux new -s redis-relay -d", shell=True)
    logger.debug(cp.stdout)
    logger.debug(cp.stderr)
    cp = subprocess.run("tmux send-keys -t redis-relay \"redis-server redis_config/redis_local_mirror.conf\" C-m", shell=True)
    logger.debug(cp.stdout)
    logger.debug(cp.stderr)


    # Load the experiment from file
    logger.info("working dir: {}".format(os.getcwd()))
    with open("configurations/skiing.json", 'r') as f:
        exp_config = json.loads(f.read())



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
            log_dir=args.log)
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
                          log_dir=args.log)
        logger.info("Node {} joining server.".format(node_id))

    print("Node {} done!".format(node_id))


