import json
#import redis
import logging
import os
#import hostlist
from nodes import MasterNode, WorkerNode
from clients import MasterClient, RelayClient, WorkerClient
import argparse

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.Logger("fake_cluster")

    parser = argparse.ArgumentParser(description='Start a master or worker node for the GA experiment')
    parser.add_argument('--master', dest='is_master', action='store_const',
                        const=True, default=False,
                        help='make this node the master')

    with open("configurations/skiing.json", 'r') as f:
        exp_config = json.loads(f.read())

    args = parser.parse_args()

    # Load the experiment from file
    if args.is_master:
        # This node contains the master
        master_node = MasterNode(
            0,
            1,
            exp_config,
            master_host='localhost',
            master_port=6379,
            relay_socket='/tmp/es_redis_relay.sock',
            master_pw= "deepbrickwindowattack",
            log_dir = "logs/login_node_test")
        master_node.begin_exp()

    else:
        # start the workers subscriptions
        node = WorkerNode(1,
                          10,
                          exp_config,
                          master_host='10.43.7.13',
                          master_port=6379,
                          relay_socket='/tmp/es_redis_relay.sock',
                          master_pw= "deepbrickwindowattack",
                          log_dir = "logs/login_node_test")
        logger.info("Node {} joining server.".format(1))

    print("done!")


