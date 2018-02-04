# ga-dist
The idea of this project is to use evolutionary algorithms to do MAML (Finn et al https://arxiv.org/abs/1703.03400), but looping over data-collection -> gradient update multiple times, rather than just doing it once. 

The evolutionary algorithm is critical, as we can't do multiple gradient updates in a principled way with a gradient-based method. It's also faster.

The GA I'm using is distributed truncation selection running on Cambridge's Peta4 cluster.

Forked from Salimans et al https://github.com/openai/evolution-strategies-starter to use truncation selection instead of evolution strategies, and to run on the Peta4 CPU cluster instead of AWS. 

Also uses `gym`, `roboschool`, `redis`.

# Explanation

Use `launch_multi.py` to launch several experiments with different random seeds and environments. A ~1bn timestep run of one Atari env takes approximately 1280 core-hours. This calls `sbatch slurm_python.peta4`.

Running `sbatch slurm_python.peta4` on a login node launches a single experiment across multiple nodes (currently 8 nodes for 5 hours). The `gym` environment id is passed as an environment (in the shell sense) variable. This slurm script asks the cluster to run `main.py` on each node.

`main.py` runs on each node. The 0th node (as allocated by SLURM) is the 'master' and the rest are 'workers'. 31 processes on each node (incl. master) asynchronously perform genome mutation and rollouts. Each node has a local 'relay' redis databases that all 31 processes push results to. The relay database batches and pushes results to the 'master' redis databse on the master node. After receiving N results, the master chooses the next generation's parents and broadcasts their genomes.

The algorithm, truncation selection, (see https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)) is run with a low bandwidth requirement using the following trick: each node generates 1GB of Gaussian noise before the experiment starts. The parameters of a neural network can be described using indices into this noise, the number of neural network parameters and a known initialization scheme. To communicate the fitness of a parameterization thus generated (a 'result') we need only send these indices and the fitness (a single sample of the return). The number of 'noise indices' is equal to the number of generations completed. Since this is on the order of 1000, each result is at most ~1kB in size. 

# Current progress

Changed the fork to use the conceptually simpler truncation selection algorithm (as per http://eng.uber.com/wp-content/uploads/2017/12/deep-ga-arxiv.pdf).

Heavy refactoring of the original `es.py` and `dist.py` into 'Node' and 'Client' classes according to my own intuition.

Added compatibility for Atari environments (DeepMind configuration).

Added more logging and inspection notebooks.

Can now successfully run many experiments at once on the Peta4 cluster!

# Next steps

Full run of 11 Atari games (10 seeds each).

Wall-clock-time vs. core-number scaling tests.

Briefly waiting for cluster changes to perform roboschool experiments.

GA-MAML on ant meta-task!

# Stuff I've learned

* Networking networking networking. How to use a Unix domain socket, TCP vs UDP. I still need to work on this though.
* Writing asynchronous code. How to spawn processes, how to use Queues to handle asynchronous communication. 
* Arg parsing
* How a high-performance compute resource manager like SLURM works
* Proper use of a step-through debugger!

