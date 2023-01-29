from PIL import Image
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import numpy as np

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

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

run_path = "/home/son/Documents/icml2023/DiffRL/icml2023_outputs/importance_sampling_differentiable_physics/cheetah"
grad_ppo_path = run_path + "/grad_ppo_backup"
ppo_path = run_path + "/ppo"

mpath = {'grad_ppo': grad_ppo_path, 'ppo': ppo_path}

steps = {}
rewards = {}

min_step = 0
max_step = 0

for method in ['grad_ppo', 'ppo']:
    mcpath = mpath[method]
    dirs = os.listdir(mcpath)

    steps[method] = {}
    rewards[method] = {}

    for dir in dirs:
        if method == 'shac':
            fpath = mcpath + "/{}".format(dir) + "/log"
        else:
            fpath = mcpath + "/{}".format(dir) + "/runs"
        out, step = tabulate_events(fpath)

        c_max_step = np.max(step)
        max_step = c_max_step if c_max_step > max_step else max_step

        steps[method][dir] = step
        rewards[method][dir] = out['rewards/step']

final_steps = {}
final_rewards = {}
final_rewards_mean = {}
final_rewards_std = {}

for method in ['grad_ppo', 'ppo']:
    c_steps = list(steps[method].values())[0]

    f_steps = []
    f_rewards = []

    for c_step in c_steps:

        # valid = True

        # for cr_step in list(steps[method].values()):

        #     if c_step not in cr_step:

        #         valid = False
        #         break

        # if not valid:
        #     continue

        f_steps.append(c_step)
        fc_rewards = []

        for cr_step_key in list(steps[method].keys()):

            cr_step = steps[method][cr_step_key]

            index = min(range(len(cr_step)), key=lambda i: abs(cr_step[i]-c_step))

            #index = cr_step.index(c_step)
            reward = rewards[method][cr_step_key][index][0]
            fc_rewards.append(reward)
        
        f_rewards.append(fc_rewards)
    
    final_steps[method] = np.array(f_steps)
    final_rewards[method] = np.array(f_rewards)
    final_rewards_mean[method] = np.mean(final_rewards[method], axis=1)
    final_rewards_std[method] = np.std(final_rewards[method], axis=1)

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
    [0, 0.45, 0.74],            # Ours
    [0.85, 0.33, 0.1],          # PPO
    [0.9290, 0.6940, 0.1250],   # SHAC
    [0.4940, 0.1840, 0.5560],   # SAC
    [0.4660, 0.6740, 0.1880]
]
with sns.axes_style("darkgrid"):

    # max_ddpg_spline = make_interp_spline(ddpg_timesteps, max_ddpg_evaluates)
    # mean_ddpg_spline = make_interp_spline(ddpg_timesteps, mean_ddpg_evaluates)
    # std_ddpg_spline = make_interp_spline(ddpg_timesteps, std_ddpg_evaluates)

    # max_ppo_spline = make_interp_spline(ddpg_timesteps, max_ppo_evaluates)
    # mean_ppo_spline = make_interp_spline(ppo_timesteps, mean_ppo_evaluates)
    # std_ppo_spline = make_interp_spline(ppo_timesteps, std_ppo_evaluates)

    # max_sac_spline = make_interp_spline(ddpg_timesteps, max_sac_evaluates)
    # mean_sac_spline = make_interp_spline(sac_timesteps, mean_sac_evaluates)
    # std_sac_spline = make_interp_spline(sac_timesteps, std_sac_evaluates)

    # max_my_spline = make_interp_spline(ddpg_timesteps, max_my_evaluates)
    # mean_my_spline = make_interp_spline(my_timesteps, mean_my_evaluates)
    # std_my_spline = make_interp_spline(my_timesteps, std_my_evaluates)

    #n_timesteps = np.linspace(min_step, max_step, 1)

    # max_ddpg_results = max_ddpg_spline(n_timesteps)
    # mean_ddpg_results = mean_ddpg_spline(n_timesteps)
    # std_ddpg_results = std_ddpg_spline(n_timesteps)

    # max_ppo_results = max_ppo_spline(n_timesteps)
    # mean_ppo_results = mean_ppo_spline(n_timesteps)
    # std_ppo_results = std_ppo_spline(n_timesteps)

    # max_sac_results = max_sac_spline(n_timesteps)
    # mean_sac_results = mean_sac_spline(n_timesteps)
    # std_sac_results = std_sac_spline(n_timesteps)

    # max_my_results = max_my_spline(n_timesteps)
    # mean_my_results = mean_my_spline(n_timesteps)
    # std_my_results = std_my_spline(n_timesteps)

    #plt.plot(n_timesteps, max_ddpg_results, c = clrs[0], linewidth=2, label="DDPG")
    n_timesteps = final_steps['grad_ppo']
    mean_rewards = final_rewards_mean['grad_ppo']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['grad_ppo']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[0], label=r"$\alpha=0.1$", linewidth=5)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[0])


    n_timesteps = final_steps['ppo']
    mean_rewards = final_rewards_mean['ppo']
    mean_rewards_spline = make_interp_spline(n_timesteps, mean_rewards)
    std_rewards = final_rewards_std['ppo']
    std_rewards_spline = make_interp_spline(n_timesteps, std_rewards)
    n_timesteps = np.linspace(min_step, max_step, 200)
    mean_rewards = mean_rewards_spline(n_timesteps)
    std_rewards = std_rewards_spline(n_timesteps)
    plt.plot(n_timesteps, mean_rewards, c = clrs[1], label="PPO", linewidth=5)
    plt.fill_between(n_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, facecolor = clrs[1])


    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.legend()

    #plt.show()
    plt.savefig(run_path + "/learning_graph.pdf")



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