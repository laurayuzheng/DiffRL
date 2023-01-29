from PIL import Image
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import numpy as np

save_path = "./"
env_name = "HalfCheetah"

alphas = ['2e-1', '4e-1']

base_paths = []
for alpha in alphas:
    base_path = "./icml2023_outputs/importance_sampling_sb3/TRPO_compare/{}/base/{}/".format(alpha, env_name)
    base_paths.append(base_path)

diff_paths = []
for alpha in alphas:
    diff_path = "./icml2023_outputs/importance_sampling_sb3/TRPO_compare/{}/diff/{}/".format(alpha, env_name)
    diff_paths.append(diff_path)

base_files = []
diff_files = []

for bp in base_paths:
    for hist in os.listdir(bp):
        base_files.append(bp + "/{}/evaluations.npz".format(hist))
for dp in diff_paths:
    curr_diff_files = []
    for hist in os.listdir(dp):
        curr_diff_files.append(dp + "/{}/evaluations.npz".format(hist))
    diff_files.append(curr_diff_files)

base_evals = [np.load(file) for file in base_files]
diff_evals = []
for curr_diff_files in diff_files:
    curr_diff_evals = [np.load(file) for file in curr_diff_files]
    diff_evals.append(curr_diff_evals)

base_timesteps = base_evals[0]['timesteps']
base_evaluates = [np.mean(e['results'], axis=1) for e in base_evals]
base_evaluates_num = min([len(e) for e in base_evaluates])
base_evaluates = [e[:base_evaluates_num] for e in base_evaluates]
base_timesteps = base_timesteps[:base_evaluates_num]

mean_base_evaluates = np.mean(base_evaluates, axis=0)
std_base_evaluates = np.std(base_evaluates, axis=0)
max_base_evaluates = np.max(base_evaluates,axis=(0, 1))

diff_timesteps = []
mean_diff_evaluates = []
std_diff_evaluates = []
max_diff_evaluates = []

for curr_diff_evals in diff_evals:
        
    curr_diff_timesteps = curr_diff_evals[0]['timesteps']
    curr_diff_evaluates = [np.mean(e['results'], axis=1) for e in curr_diff_evals]
    curr_diff_evaluates_num = min([len(e) for e in curr_diff_evaluates])
    curr_diff_evaluates = [e[:curr_diff_evaluates_num] for e in curr_diff_evaluates]
    curr_diff_timesteps = curr_diff_timesteps[:curr_diff_evaluates_num]

    curr_mean_diff_evaluates = np.mean(curr_diff_evaluates, axis=0)
    curr_std_diff_evaluates = np.std(curr_diff_evaluates, axis=0)
    curr_max_diff_evaluates = np.max(curr_diff_evaluates,axis=(0, 1))

    diff_timesteps.append(curr_diff_timesteps)
    mean_diff_evaluates.append(curr_mean_diff_evaluates)
    std_diff_evaluates.append(curr_std_diff_evaluates)
    max_diff_evaluates.append(curr_max_diff_evaluates)

params = {'legend.fontsize': 50,
          'figure.figsize': (12 * 1.5, 9 * 1.5),
         'axes.labelsize': 30,
         'axes.titlesize': 30,
         'xtick.labelsize':30,
         'ytick.labelsize':30}
#fig.set_figwidth(12)
#fig.set_figheight(9)

plt.rcParams.update(params)
plt.grid(alpha=0.3)
clrs = [
    [0, 0, 0],            # BASE
    [0.85, 0.33, 0.1],          # OURS
    [0.9290, 0.6940, 0.1250],   # OURS
    [0.4940, 0.1840, 0.5560],   # OURS
]
with sns.axes_style("darkgrid"):
    plt.plot(base_timesteps, mean_base_evaluates, label="PPO", c = clrs[0], linewidth=5)
    plt.fill_between(base_timesteps, mean_base_evaluates - std_base_evaluates * 0.3, mean_base_evaluates + std_base_evaluates * 0.3, alpha=0.3, facecolor = clrs[0])

    for i in range(len(alphas)):
        alpha = alphas[i]
        curr_diff_timesteps = diff_timesteps[i]
        curr_mean_diff_evaluates = mean_diff_evaluates[i]
        curr_std_diff_evaluates = std_diff_evaluates[i]

        plt.plot(curr_diff_timesteps, curr_mean_diff_evaluates, label=r"$\alpha = {}$".format(alpha), c = clrs[1 + i], linewidth=5)
        plt.fill_between(curr_diff_timesteps, curr_mean_diff_evaluates - curr_std_diff_evaluates * 0.3, curr_mean_diff_evaluates + curr_std_diff_evaluates * 0.3, alpha=0.3, facecolor = clrs[1 + i])

    plt.legend()

    plt.xlabel("Step")
    plt.ylabel("Reward")

    #plt.title("Env: {} / RL Algorithm: {}".format(env_name, "TRPO"), fontsize=20)

    #plt.show()
    plt.savefig(save_path + "/learning_graph.pdf")



'''
for path in my_path:
    file = open(path)
    line = file.readline()
    words = line.split()
    npath = path + "n"
    nfile = open(npath, 'w')
    # epoch
    cnt = 0
    epoch = 1
    while True:
        episode = int(words[cnt + 1])
        step = int(words[cnt + 2])
        reward = float(words[cnt + 3][:-len(str(epoch + 1))])
        nfile.write("{} {} {} {}\n".format(epoch, episode, step, reward))
        cnt = cnt + 3
        epoch += 1
        if cnt + 1 >= len(words):
            break
    file.close()
    nfile.close()
exit()
'''