# start the master redis server
import os
import subprocess
if os.environ["SLURM_NODEID"]== 0:
    subprocess.run(["tmux", "new", "-s", "redis-master", "-d"])
    subprocess.run(["tmux", "send-keys", "-t", "redis-master", "redis-server redis_config/redis_master.conf", "C-m"])
# start the relay redis server
subprocess.run(["tmux", "new", "-s", "redis-relay", "-d"])
subprocess.run(["tmux", "send-keys", "-t", "redis-relay", "redis-server redis_config/redis_local_mirror.conf", "C-m"])

# run the script
python ga_dist/main.py --log $1