# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time

import torch
import random

import envs
from utils.common import *

import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type = str, default = 'TrafficRoundaboutEnv')
parser.add_argument('--num-envs', type = int, default = 1)
parser.add_argument('--render', default = False, action = 'store_true')

args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

seeding()

env_fn = getattr(envs, args.env)

env = env_fn(num_envs = args.num_envs, \
            device = device, \
            render = args.render, \
            seed = 0, \
            stochastic_init = True, \
            MM_caching_frequency = 16, \
            no_grad = True, \
            no_steering = True, \
            num_idm_vehicle = 2, \
            num_auto_vehicle = 0, \
            # num_lane = 4
        )

obs = env.reset()

num_actions = env.num_actions

t_start = time.time()

reward_episode = 0.
for i in range(1500):
    actions = torch.zeros((args.num_envs, num_actions), device=device)
    # actions = torch.Tensor([0]).expand(args.num_envs, -1).to(device)
    obs, reward, done, info = env.step(actions)
    reward_episode += reward

t_end = time.time()

print("Total time: %f s", (t_end - t_start))
print("Steps / second: ", 1500 / (t_end - t_start))

print('fps = ', 1000 * args.num_envs / (t_end - t_start))
print('mean reward = ', reward_episode.mean().detach().cpu().item())

print('Finish Successfully')

