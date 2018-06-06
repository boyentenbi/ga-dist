# ga-dist

The GA I'm using is distributed truncation selection running on Cambridge's Peta4 cluster.

Forked from Salimans et al https://github.com/openai/evolution-strategies-starter to use truncation selection instead of evolution strategies, and to run on the Peta4 CPU cluster instead of AWS. 

Also uses `gym`, `roboschool`, `redis`.

# Stuff I've learned

* Networking networking networking. How to use a Unix domain socket, TCP vs UDP. I still need to work on this though.
* Writing asynchronous code. How to spawn processes, how to use Queues to handle asynchronous communication. 
* Arg parsing
* How a high-performance compute resource manager like SLURM works
* Proper use of a step-through debugger!
* Practicing OO design

# Code explanation

Use `launch_multi.py` to launch several experiments with different random seeds and environments. A ~1bn timestep run of one Atari env takes approximately 1280 core-hours. This calls `sbatch slurm_python.peta4`.

Running `sbatch slurm_python.peta4` on a login node launches a single experiment across multiple nodes (currently 8 nodes for 5 hours). The `gym` environment id is passed as an environment (in the shell sense) variable. This slurm script asks the cluster to run `main.py` on each node.

`main.py` runs on each node. The 0th node (as allocated by SLURM) is the 'master' and the rest are 'workers'. 31 processes on each node (incl. master) asynchronously perform genome mutation and rollouts. Each node has a local 'relay' redis databases that all 31 processes push results to. The relay database batches and pushes results to the 'master' redis databse on the master node. After receiving N results, the master chooses the next generation's parents and broadcasts their genomes.

The algorithm, truncation selection, (see https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)) is run with a low bandwidth requirement using the following trick: each node generates 1GB of Gaussian noise before the experiment starts. The parameters of a neural network can be described using indices into this noise, the number of neural network parameters and a known initialization scheme. To communicate the fitness of a parameterization thus generated (a 'result') we need only send these indices and the fitness (a single sample of the return). The number of 'noise indices' is equal to the number of generations completed. Since this is on the order of 1000, each result is at most ~1kB in size. 

# gifs!
Atari:
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/atlantisshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/kangarooshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/seaquestshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/asteroidsshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/asterixshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/assaultshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/frostbiteshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/enduroshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/Skiing-1--6520.0-37476.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/zaxxonshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/ventureshort.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/Gravitar-3-950.0-15315.gif)
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/Amidar-0-0-373.0.gif)

Meta-RL: 
![Alt Text](https://github.com/boyentenbi/ga-dist/blob/master/saccadeshort.gif)


