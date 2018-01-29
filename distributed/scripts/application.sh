# start the master redis server
if (($SLURM_NODEID ==  0)); then
    tmux new -s redis-master -d
    tmux send-keys -t redis-master "redis-server redis_config/redis_master.conf" C-m
fi

# start the relay redis server
tmux new -s redis-relay -d
tmux send-keys -t redis-relay "redis-server redis_config/redis_local_mirror.conf" C-m

# run the script
python ga_dist/main.py --log $1
