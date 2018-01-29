log config nodes mins

import os
import subprocess

os.chdir("../slurm")

n_mins = 30
configs = ["skiing.json"]

for config in configs:
    subprocess.run(["sbatch", "submit_w_args.peta4", config, str(n_mins)])

