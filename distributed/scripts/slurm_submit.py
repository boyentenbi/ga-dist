
import os
import logging
import time
import socket

import tensorflow as tf
import hostlist # Python-hostlist


# SLURM will automatically allocate a node
hostname = socket.gethostname()
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("slurm_submit")

fh = logging.FileHandler('host-{}.log'.format(hostname))
fh.setLevel(logging.DEBUG)
log.addHandler(fh)




class DistributedTF(object):

    def __init__(self):
        self.node_name = None
        self.node_list = None
        self.node_id = None
        self.num_nodes = None
        self.job_name = "ga_dist_test"
        self.setup_cluster_details()

        if self.node_id == 0:
            log.info("main node sleeping.")
            time.sleep(20)
            # just makes sure that the lead node sleeps for a while to let the others setup.
            # could actually get them to communicate so that this starts when it knows others
            # are ready.

    def setup_cluster_details(self):
        #print(os.environ)
        self.node_name = os.environ['SLURMD_NODENAME']
        self.node_list = self._parse_node_list(os.environ['SLURM_JOB_NODELIST'])
        print(self.node_list)
        self.num_tasks = len(self.node_list)  #int(os.environ["SLURM_NTASKS"])
        #assert int(self.num_tasks) == len(self.node_list), "currently only setup for one task per node"
        self.node_id = self.node_list.index(self.node_name)

        log.info("node name: {}, assigned id: {}, num tasks: {} \n node list: {}".format(
            self.node_name, self.node_id,self.num_tasks, self.node_list
        ))

    def _parse_node_list(self, node_list):
        return hostlist.expand_hostlist(node_list)

    def main(self):
        cluster = tf.train.ClusterSpec({self.job_name: ["{}:2222".format(node_name) for node_name
                                                         in self.node_list]})
        server = tf.train.Server(cluster, job_name=self.job_name, task_index=self.node_id)
        if self.node_id > 0:
            log.info("Node {} joining server.".format(self.node_id))
        else:
            log.info("Node {} defining task.".format(self.node_id))



if __name__ == '__main__':
    dtf = DistributedTF()
    dtf.main()
    print("done!")
