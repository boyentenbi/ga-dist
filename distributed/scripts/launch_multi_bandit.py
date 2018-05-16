import os
import subprocess
import time
import numpy as np

import itertools

super_exp_id = time.strftime("%Y:%m:%d-%H:%M:%S")
# grid = {
#     "noise_stdev": [0.003, 0.01, 0.03],
#     "episodes_per_batch": [3000, 10000, 30000],
#     "n_parents": [30, 100, 300],
#     "n_blocks": [1, 2, 3],
# }
#
#
# combs = [()]
#
# for k, v in grid.items():
#     new_combs = []
#     for comb in combs:
#         for p in v:
#             new_combs.append(comb+(p,))
#     combs = new_combs
#
# len(combs)

NKs = [(1000, 50), (500,50), (500,10), (500,5), (100,50), (100,10), (100,5), (10,50), (10,10), (10,5)]

seeds = range(5)
rnd = np.random.randint(2**31)

if not input("Is your config file correct? y/n:")=="y":
    exit()
if not input("Is the number of hours and nodes and tasks set correctly in your slurm script? y/n:")=="y":
    exit()
print("If you wish to proceed, copy the following number: {}".format(rnd))
x=input(">>")
if x == str(rnd):

    os.environ["super_exp_id"]=super_exp_id
    os.mkdir(os.path.join("logs", super_exp_id))
    for env_id in NKs:
        if not env_id =="-":
            os.mkdir(os.path.join("logs", super_exp_id, env_id))

            for seed in seeds:
                os.mkdir(os.path.join("logs", super_exp_id, env_id, str(seed)))
                new_shell_env = os.environ.copy()
                new_shell_env["global_seed"]=str(seed)
                new_shell_env["env_id"]=env_id
                subprocess.run("sbatch slurm/slurm_python_single.peta4".split(),env=new_shell_env)

else:
    print("Closing...")
