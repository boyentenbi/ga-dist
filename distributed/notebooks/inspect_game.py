import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import numpy as np
import os
import matplotlib
super_exp_path = '/home/pc517/ga-dist/distributed/logs/2018:05:12-11:43:12'
print("Loading results files. This may take a while...")
print("Number of results files found in {}:".format(super_exp_path))
dfs = {}

matplotlib.rcParams.update({'font.size': 6})

test_mode = False

n_tsteps = 250000000
tick = 10 ** 7
frames_per_step = 4
n_cols = 3
n_games = 13
n_xticks = 5
n_rows = int(np.ceil(n_games / n_cols))
game_counter = 0
graph_len = n_tsteps*frames_per_step / tick + 1
fig, axs = plt.subplots(ncols=n_cols, nrows = n_rows, figsize=(7,9))

if test_mode:
    envs = ["AsteroidsNoFrameskip-v4"] # os.listdir(super_exp_path)
else:
    envs = os.listdir(super_exp_path)

for env_id in envs:
    env_exp_path = os.path.join(super_exp_path, env_id)

    seed_count = 0
    dfs[env_id] = {}
    for seed in os.listdir(env_exp_path):
        results_path = os.path.join(env_exp_path, seed, "results.csv")
        if not os.path.isfile(results_path):
            continue
        results = pandas.read_csv(results_path)
        seed_count += 1

        dfs[env_id][seed] = results
        if test_mode:
            break
    print("    {}: {}".format(env_id, seed_count))

print("Generating graph data")

game_counter = 0
for env_id, env_results in dfs.items():
    graphs = []
    for seed, seed_results in env_results.items():

        # get all valid evals
        valid_results = seed_results[lambda x: x["is_valid"]]
        # Get generation finish time

        # print([(gen_num, len(gen_data)) for gen_num, gen_data in grouped])
        gen_graph = []
        gen_frames = []

        ys = []
        gen_zero_data = valid_results[valid_results["worker_gen"] == 0]
        gen_zero_frames = gen_zero_data["n_steps"].sum() * frames_per_step
        xs = [i * tick for i in range(int(np.floor(gen_zero_frames / tick)) + 1)]
        ys = [None for x in xs]

        t = gen_zero_frames

        # Group into generations
        grouped = valid_results[valid_results["worker_gen"] > 0].groupby("worker_gen")

        grouped.apply(pandas.DataFrame.sort_values, 'worker_gen')
        n_gens = len(grouped.groups.keys())+1
        # print("Got {} gens for {}/{}".format(n_gens, env_id, seed))
        assert list(grouped.groups.keys())== [x for x in range(1, n_gens)]
        for gen_num, gen_data in grouped:

            n_steps = gen_data["n_steps"].sum()
            n_frames = n_steps * frames_per_step
            t += n_frames

            # These evals correspond to candidates from previous gen
            # So it is correct to set the new xs to the elite value found here
            evals = gen_data[gen_data["is_eval"]]
            candidate_means = evals[["noise_list", "ret"]].groupby("noise_list").mean()
            elite_mean = candidate_means.max()

            while xs[-1] + tick <= t:
                xs.append(xs[-1] + tick)
                ys.append(elite_mean)

        graphs.append(ys)
    # for g in graphs:
    #     assert len(g) == graph_len
    starts = []
    for g in graphs:
        starts.append(min([i for i in range(len(g)) if g[i] is not None]))
    #     print("graph has len {}, start {}".format(len(g), starts[-1]))
    start = max(starts)
    end = min([len(g) for g in graphs])
    graphs_trunc = np.asarray([g[start:end] for g in graphs]).squeeze(-1)
    ax=axs[int(np.floor(game_counter/n_cols)), int(game_counter%n_cols)]
    sns.tsplot(graphs_trunc, xs[start:end], estimator=np.median, ci=95, ax=ax)
    plt.setp(ax.lines, linewidth=0.6)
    ax.set_title("{}".format(env_id[:-14]))
    game_counter +=1

# remove all xticks
for i in range(n_games):
    col = int(i%n_cols)
    row = int(np.floor(i/n_cols))
    ax=axs[row, col]
    ax.set_xticks([])

# add xticks to bottom plots
for i in range(n_games-n_cols,n_games):
    col = int(i%n_cols)
    row = int(np.floor(i/n_cols))
    ax=axs[row, col]
    ax.set_xlabel("Number of game frames", labelpad=5)
    ax.set_xticks([n_tsteps * frames_per_step / n_xticks * i for i in range(n_xticks+1)])
    # t = ax.xaxis.get_offset_text()
    # t.set_size(5)
    # ax.set_xticklabels(
    #         [int(n_tsteps * frames_per_step / n_xticks  * i)\
    #          for i in range(n_xticks+1)])
    ax.ticklabel_format(style="sci", useMathText=False)
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # ax.set_xlim(0,10**9)

fig.subplots_adjust(wspace=0.3, hspace=0.3)
for i in range(n_games,n_cols*n_rows):

    col = int(i%n_cols)
    row = int(np.floor(i/n_cols))
    fig.delaxes(axs[row, col])
plt.savefig('/home/pc517/ga-dist/distributed/notebooks/learning_curves.pdf', bbox_inches='tight')
