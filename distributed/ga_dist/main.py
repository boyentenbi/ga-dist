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
    with open("../configurations/cartpole.json", 'r') as f:
        exp_config = json.loads(f.read())


    parser = argparse.ArgumentParser(description='Start a master or worker node for the GA experiment')
    parser.add_argument('--master', dest='is_master', action='store_const',
                        const=True, default=False,
                        help='make this node the master')

    args = parser.parse_args()

    # Load the experiment from file
    if args.is_master:
        # This node contains the master
        master_node = MasterNode(
            0,
            0,
            exp_config,
            master_host='localhost',
            master_port=6379,
            relay_socket='/tmp/es_redis_relay.sock',
            master_pw= "deepbrickwindowattack")
        master_node.begin_exp('logs')

    else:
        # start the workers subscriptions
        node = WorkerNode(1,
                          1,
                          exp_config,
                          master_host='10.43.7.13',
                          master_port=6379,
                          relay_socket='/tmp/es_redis_relay.sock',
                          master_pw= "deepbrickwindowattack")
        logger.info("Node {} joining server.".format(1))

    print("done!")


