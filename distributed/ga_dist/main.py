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
ATARI_ENV_IDS = [
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "AsteroidsNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "BerzerkNoFrameskip-v4",
    "BowlingNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "ChopperCommandNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "-",
    "DemonAttackNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "GravitarNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "IceHockeyNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MontezumaRevengeNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "PitfallNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "PrivateEyeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "RobotankNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SkiingNoFrameskip-v4",
    "SolarisNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "-",
    "TennisNoFrameskip-v4",
    "TimePilotNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "VentureNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4",
    "WizardOfWorNoFrameskip-v4",
    "YarsRevengeNoFrameskip-v4",
    "ZaxxonNoFrameskip-v4",]

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
    array_id = os.environ["SLURM_ARRAY_TASK_ID"]
    with open(args.config, 'r') as f:
        exp_config = json.loads(f.read())
        exp_config["env_id"] = ATARI_ENV_IDS[array_id]
    logger.info("Starting GA. array_id = {}, env_id = {}".format(array_id, exp_config["env_id"]))

    env_log_dir = os.path.join(args.log, exp_config["env_id"])

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
            log_dir=env_log_dir)
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
                          log_dir=env_log_dir)
        logger.info("Node {} joining server.".format(node_id))

    print("Node {} done!".format(node_id))


