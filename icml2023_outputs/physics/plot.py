from PIL import Image
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import numpy as np

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    summary_iterators = []
    for dname in os.listdir(dpath):
        if 'events' in dname:
            summary_iterators.append(EventAccumulator(os.path.join(dpath, dname)).Reload())

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    #for tag in tags:
    tag = "rewards/step"
    steps = [e.step for e in summary_iterators[0].Scalars(tag)]

    for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
        assert len(set(e.step for e in events)) == 1

        out[tag].append([e.value for e in events])

    return out, steps


'''
==========================================
'''

run_path = "/home/son/Documents/icml2023/DiffRL/icml2023_outputs/physics/cheetah"
grad_ppo_0_path = run_path + "/grad_ppo_0"
grad_ppo_1e_4_path = run_path + "/grad_ppo_1e-4"
grad_ppo_5e_4_path = run_path + "/grad_ppo_5e-4"
grad_ppo_1e_3_path = run_path + "/grad_ppo_1e-3"
grad_ppo_5e_3_path = run_path + "/grad_ppo_5e-3"
shac_path = run_path + "/shac"

mpath = {'grad_ppo_0': grad_ppo_0_path, 
        'grad_ppo_1e_4': grad_ppo_1e_4_path, 
        'grad_ppo_5e_4': grad_ppo_5e_4_path, 
        'grad_ppo_1e_3': grad_ppo_1e_3_path, 
        'grad_ppo_5e_3': grad_ppo_5e_3_path, 
        'shac': shac_path}

steps = {}
rewards = {}

min_step = 0
max_step = 0

for method in mpath.keys():
    mcpath = mpath[method]
    dirs = os.listdir(mcpath)

    steps[method] = {}
    rewards[method] = {}

    for dir in dirs:
        if method == 'shac':
            fpath = mcpath + "/{}".format(dir)
        else:
            fpath = mcpath + "/{}".format(dir)
        out, step = tabulate_events(fpath)

        c_max_step = np.max(step)
        max_step = c_max_step if c_max_step > max_step else max_step

        steps[method][dir] = step
        rewards[method][dir] = out['rewards/step']

final_steps = {}
final_rewards = {}
final_rewards_mean = {}
final_rewards_std = {}

for method in mpath.keys():
    c_steps = list(steps[method].values())[0]

    f_steps = []
    f_rewards = []

    for c_step in c_steps:

        valid = True

        for cr_step in list(steps[method].values()):

            if c_step not in cr_step:

                valid = False
                break

        if not valid:
            continue

        f_steps.append(c_step)
        fc_rewards = []

        for cr_step_key in list(steps[method].keys()):

            cr_step = steps[method][cr_step_key]
            index = cr_step.index(c_step)
            reward = rewards[method][cr_step_key][index][0]
            fc_rewards.append(reward)
        
        f_rewards.append(fc_rewards)
    
    final_steps[method] = np.array(f_steps)
    final_rewards[method] = np.array(f_rewards)
    final_rewards_mean[method] = np.mean(final_rewards[method], axis=1)
    final_rewards_std[method] = np.std(final_rewards[method], axis=1)

params = {'legend.fontsize': 25,
          'figure.figsize': (12, 9),
         'axes.labelsize': 30,
         'axes.titlesize': 30,
         'xtick.labelsize':30,
         'ytick.labelsize':30}
#fig.set_figwidth(12)
#fig.set_figheight(9)

plt.rcParams.update(params)
plt.grid(alpha=0.3)
clrs = [
    [0, 0.45, 0.74],            # Ours (0)
    [0.85, 0.33, 0.1],          # Ours (1)
    [0.9290, 0.6940, 0.1250],   # Ours (2)
    [0.4940, 0.1840, 0.5560],   # Ours (3)
    [0.4660, 0.6740, 0.1880],   # Ours (4)
    [0, 0, 0],                  # SHAC
]
with sns.axes_style("darkgrid"):
    n_timesteps = final_steps['grad_ppo_0']
    mean_rewards = final_rewards_mean['grad_ppo_0']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['grad_ppo_0']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[0], label="Ours(GradPPO, 0)", linewidth=3)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[0])

    n_timesteps = final_steps['grad_ppo_1e_4']
    mean_rewards = final_rewards_mean['grad_ppo_1e_4']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['grad_ppo_1e_4']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[1], label="Ours(GradPPO, 1e-4)", linewidth=3)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[1])

    n_timesteps = final_steps['grad_ppo_5e_4']
    mean_rewards = final_rewards_mean['grad_ppo_5e_4']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['grad_ppo_5e_4']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[2], label="Ours(GradPPO, 5e-4)", linewidth=3)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[2])

    n_timesteps = final_steps['grad_ppo_1e_3']
    mean_rewards = final_rewards_mean['grad_ppo_1e_3']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['grad_ppo_1e_3']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[3], label="Ours(GradPPO, 1e-3)", linewidth=3)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[3])

    n_timesteps = final_steps['grad_ppo_5e_3']
    mean_rewards = final_rewards_mean['grad_ppo_5e_3']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['grad_ppo_5e_3']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[4], label="Ours(GradPPO, 5e-3)", linewidth=3)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[4])

    n_timesteps = final_steps['shac']
    mean_rewards = final_rewards_mean['shac']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['shac']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[5], label="SHAC", linewidth=3)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[5])

    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.legend()

    #plt.show()
    plt.savefig(run_path + "/learning_graph.png")



# '''
# for path in my_path:
#     file = open(path)
#     line = file.readline()
#     words = line.split()
#     npath = path + "n"
#     nfile = open(npath, 'w')
#     # epoch
#     cnt = 0
#     epoch = 1
#     while True:
#         episode = int(words[cnt + 1])
#         step = int(words[cnt + 2])
#         reward = float(words[cnt + 3][:-len(str(epoch + 1))])
#         nfile.write("{} {} {} {}\n".format(epoch, episode, step, reward))
#         cnt = cnt + 3
#         epoch += 1
#         if cnt + 1 >= len(words):
#             break
#     file.close()
#     nfile.close()
# exit()
# '''