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

# NKs = [(1000, 50), (500,50), (500,10), (500,5), (100,50),
#        (100,10), (100,5), (10,50), (10,10), (10,5)]
NKs = [(500,50)]
NKs.reverse()

seeds = range(1)
rnd = np.random.randint(2**31)

if not input("Is your config file correct? y/n:")=="y":
    exit()
if not input("Is the number of hours and nodes and tasks set correctly in your slurm script? y/n:")=="y":
    exit()
print("If you wish to proceed, copy the following number: {}".format(rnd))
x=input(">>")

env_id = "MetaBandit-v0"
if x == str(rnd):

    os.environ["super_exp_id"]=super_exp_id
    os.mkdir(os.path.join("logs", super_exp_id))
    os.mkdir(os.path.join("logs", super_exp_id, env_id))
    for N,K in NKs:

        os.mkdir(os.path.join("logs", super_exp_id, env_id, "{}-{}".format(N,K)))

        for seed in seeds:
            os.mkdir(os.path.join("logs", super_exp_id, env_id, "{}-{}".format(N,K),
                                  str(seed)))
            new_shell_env = os.environ.copy()
            new_shell_env["global_seed"]=str(seed)
            new_shell_env["env_id"]=env_id
            new_shell_env["n_eps"]= str(N)
            new_shell_env["n_levers"] = str(K)
            new_shell_env["config"] = "meta_bandit.json"
            subprocess.run("sbatch slurm/slurm_meta_bandit_multi.peta4-knl".split(),env=new_shell_env)

else:
    print("Closing...")
