import json
import redis
import logging
import os
import hostlist
logging.basicConfig(level=logging.DEBUG)
log = logging.Logger("fake_cluster")
from .nodes import MasterNode, WorkerNode
from .clients import MasterClient, RelayClient, WorkerClient


with open("configurations/cartpole.json", 'r') as f:
    exp_config = json.loads(f.read())

import time
# Simulate server starts

class DistributedTF(object):

    def __init__(self, num_nodes, node_id):
        self.node_name = None
        self.node_list = None
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.job_name = "ga_dist_test"
        self.fake_setup()


        if self.node_id == 0:
            log.info("main node sleeping.")
            time.sleep(1)
            # just makes sure that the lead node sleeps for a while to let the others setup.
            # could actually get them to communicate so that this starts when it knows others
            # are ready.

    def fake_setup(self):
        # print(os.environ)
        self.node_name = 'cpu-e-{}'.format(self.node_id)
        self.node_list = ['cpu-e-{}'.format(nid) for nid in range(self.num_nodes)]
        print(self.node_list)
        self.num_tasks = len(self.node_list)  # int(os.environ["SLURM_NTASKS"])
        # assert int(self.nawum_tasks) == len(self.node_list), "currently only setup for one task per node"
        #self.node_id = self.node_list.index(self.node_name)

        log.info("node name: {}, assigned id: {}, num tasks: {} \n node list: {}".format(
            self.node_name, self.node_id, self.num_tasks, self.node_list
        ))
        self.num_nodes = len(self.node_list)

    # def _parse_node_list(self, node_list):
    #     return hostlist.expand_hostlist(node_list)

    def main(self):

        # Load the experiment from file


        if self.node_id == 0:
            # This node contains the master
            node = MasterNode(exp_config)

        else:
            # start the workers subscriptions
            node = WorkerNode(exp_config)
            log.info("Node {} joining server.".format(self.node_id))

# Test with only the node containing the master client
N_SIM_WORKERS = 0

dtf = DistributedTF(0, node_id = 1)
dtf.main()
print("done!")


